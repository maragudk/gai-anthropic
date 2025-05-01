package anthropic

import (
	"context"
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

	if req.Messages[len(req.Messages)-1].Role != gai.MessageRoleUser {
		panic("last message must have user role")
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
			default:
				panic("not implemented")
			}
		}

		messages = append(messages, anthropic.MessageParam{
			Content: parts,
			Role:    anthropic.MessageParamRole(m.Role),
		})
	}

	// TODO: Temperature ranges from 0 to 1, normalize
	var temperature param.Opt[float64]
	if req.Temperature != nil {
		temperature = param.NewOpt[float64](req.Temperature.Float64())
	}

	stream := c.Client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		MaxTokens:   1024,
		Messages:    messages,
		Model:       string(c.model),
		Temperature: temperature,
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
			err := message.Accumulate(event)
			if err != nil {
				yield(gai.MessagePart{}, err)
				return
			}

			switch eventVariant := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				switch deltaVariant := eventVariant.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if !yield(gai.TextMessagePart(deltaVariant.Text), nil) {
						return
					}
				}
			}
		}

		if stream.Err() != nil {
			yield(gai.MessagePart{}, stream.Err())
		}
	}), nil
}
