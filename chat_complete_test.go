package anthropic_test

import (
	"testing"

	"maragu.dev/gai"
	"maragu.dev/is"

	anthropic "maragu.dev/gai-anthropic"
)

func TestChatCompleter_ChatComplete(t *testing.T) {
	t.Run("can chat-complete", func(t *testing.T) {
		c := newClient()

		cc := c.NewChatCompleter(anthropic.NewChatCompleterOptions{
			Model: anthropic.ChatCompleteModelClaude3_5HaikuLatest,
		})

		req := gai.ChatCompleteRequest{
			Messages: []gai.Message{
				gai.NewUserTextMessage("Hi!"),
			},
			Temperature: gai.Ptr(gai.Temperature(0)),
		}

		res, err := cc.ChatComplete(t.Context(), req)
		is.NotError(t, err)

		var output string
		for part, err := range res.Parts() {
			is.NotError(t, err)
			output += part.Text()
		}
		is.Equal(t, "Hello! How are you doing today? Is there anything I can help you with?", output)
	})
}
