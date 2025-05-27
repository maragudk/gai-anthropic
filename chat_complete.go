package anthropic

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"maragu.dev/gai"
)

type ChatCompleteModel string

const (
	ChatCompleteModelClaude3_5HaikuLatest  = ChatCompleteModel(anthropic.ModelClaude3_5HaikuLatest)
	ChatCompleteModelClaude3_7SonnetLatest = ChatCompleteModel(anthropic.ModelClaude3_7SonnetLatest)
	ChatCompleteModelClaude4OpusLatest     = ChatCompleteModel(anthropic.ModelClaude4Opus20250514)
	ChatCompleteModelClaude4SonnetLatest   = ChatCompleteModel(anthropic.ModelClaude4Sonnet20250514)
)

type ChatCompleter struct {
	Client anthropic.Client
	log    *slog.Logger
	model  ChatCompleteModel
}

type NewChatCompleterOptions struct {
	Model ChatCompleteModel
}

func (c *Client) NewChatCompleter(opts NewChatCompleterOptions) *ChatCompleter {
	return &ChatCompleter{
		Client: c.Client,
		log:    c.log,
		model:  opts.Model,
	}
}

// ChatComplete satisfies [gai.ChatCompleter].
func (c *ChatCompleter) ChatComplete(ctx context.Context, req gai.ChatCompleteRequest) (gai.ChatCompleteResponse, error) {
	if len(req.Messages) == 0 {
		panic("no messages")
	}

	var messages []anthropic.MessageParam
	for _, m := range req.Messages {
		var parts []anthropic.ContentBlockParamUnion

		for _, part := range m.Parts {
			switch part.Type {
			case gai.MessagePartTypeText:
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfText: &anthropic.TextBlockParam{
						Text: part.Text(),
					},
				})

			case gai.MessagePartTypeToolCall:
				toolCall := part.ToolCall()
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfToolUse: &anthropic.ToolUseBlockParam{
						ID:    toolCall.ID,
						Name:  toolCall.Name,
						Input: toolCall.Args,
					},
				})

			case gai.MessagePartTypeToolResult:
				toolResult := part.ToolResult()
				content := toolResult.Content
				var isError bool
				if toolResult.Err != nil {
					isError = true
					content = toolResult.Err.Error()
				}
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfToolResult: &anthropic.ToolResultBlockParam{
						ToolUseID: toolResult.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{
								OfText: &anthropic.TextBlockParam{
									Text: content,
								},
							},
						},
						IsError: anthropic.Bool(isError),
					},
				})

			default:
				panic("not implemented")
			}
		}

		var role anthropic.MessageParamRole
		switch m.Role {
		case gai.MessageRoleUser:
			role = anthropic.MessageParamRoleUser
		case gai.MessageRoleModel:
			role = anthropic.MessageParamRoleAssistant
		default:
			panic("unknown role " + m.Role)
		}

		messages = append(messages, anthropic.MessageParam{
			Content: parts,
			Role:    role,
		})
	}

	var tools []anthropic.ToolUnionParam
	for _, tool := range req.Tools {
		tools = append(tools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: tool.Schema.Properties,
				},
			},
		})
	}

	// TODO: Temperature ranges from 0 to 1, normalize
	var temperature param.Opt[float64]
	if req.Temperature != nil {
		temperature = param.NewOpt(req.Temperature.Float64())
	}

	var system []anthropic.TextBlockParam
	if req.System != nil {
		system = []anthropic.TextBlockParam{
			{
				Text: *req.System,
			},
		}
	}

	stream := c.Client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		MaxTokens:   1024, // TODO make variable
		Messages:    messages,
		Model:       anthropic.Model(c.model),
		System:      system,
		Temperature: temperature,
		Tools:       tools,
	})

	return gai.NewChatCompleteResponse(func(yield func(gai.MessagePart, error) bool) {
		defer func() {
			if err := stream.Close(); err != nil {
				c.log.Info("Error closing stream", "error", err)
			}
		}()

		var message anthropic.Message
		for stream.Next() {
			event := stream.Current()

			if err := message.Accumulate(event); err != nil {
				// A hack to circumvent a bug, see https://github.com/anthropics/anthropic-sdk-go/issues/164
				if !strings.Contains(err.Error(), "unexpected end of JSON input") {
					yield(gai.MessagePart{}, fmt.Errorf("error accumulating message: %w", err))
					return
				}
			}

			switch event := event.AsAny().(type) {
			case anthropic.ContentBlockStartEvent:

			case anthropic.ContentBlockDeltaEvent:
				switch delta := event.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if !yield(gai.TextMessagePart(delta.Text), nil) {
						return
					}
				}

			case anthropic.ContentBlockStopEvent:
				// Use the accumulated block for the tool use only
				for _, block := range message.Content {
					switch block := block.AsAny().(type) {
					case anthropic.ToolUseBlock:
						c.log.Debug("Tool call", "id", block.ID, "name", block.Name, "input", block.Input)
						var found bool
						for _, tool := range req.Tools {
							if tool.Name == block.Name {
								found = true
								if !yield(gai.ToolCallPart(block.ID, block.Name, block.Input), nil) {
									return
								}
							}
						}
						if !found {
							panic(fmt.Errorf("tool not found: %s", block.Name)) // TODO
						}
					}
				}
				message = anthropic.Message{}
			}
		}

		if stream.Err() != nil {
			yield(gai.MessagePart{}, stream.Err())
		}
	}), nil
}

var _ gai.ChatCompleter = (*ChatCompleter)(nil)
