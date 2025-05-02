package anthropic

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"maragu.dev/gai"
)

type ChatCompleteModel string

const (
	ChatCompleteModelClaude3_5HaikuLatest  = ChatCompleteModel(anthropic.ModelClaude3_5HaikuLatest)
	ChatCompleteModelClaude3_7SonnetLatest = ChatCompleteModel(anthropic.ModelClaude3_7SonnetLatest)
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
					OfRequestTextBlock: &anthropic.TextBlockParam{
						Text: part.Text(),
					},
				})

			case gai.MessagePartTypeToolCall:
				toolCall := part.ToolCall()
				parts = append(parts, anthropic.ContentBlockParamUnion{
					OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
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
					OfRequestToolResultBlock: &anthropic.ToolResultBlockParam{
						ToolUseID: toolResult.ID,
						Content: []anthropic.ToolResultBlockParamContentUnion{
							{
								OfRequestTextBlock: &anthropic.TextBlockParam{
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

		messages = append(messages, anthropic.MessageParam{
			Content: parts,
			Role:    anthropic.MessageParamRole(m.Role),
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
		temperature = param.NewOpt[float64](req.Temperature.Float64())
	}

	stream := c.Client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		MaxTokens:   1024, // TODO make variable
		Messages:    messages,
		Model:       string(c.model),
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
				yield(gai.MessagePart{}, fmt.Errorf("error accumulating message: %w", err))
				return
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
