import gradio as gr
from vector_search import HybridVectorSearch
from whisper_asr import WhisperAutomaticSpeechRecognizer

with gr.Blocks() as demo:
    with gr.Tab("Live Mode"):
        full_stream = gr.State()
        transcript = gr.State(value="")
        chats = gr.State(value=[])

        with gr.Row(variant="panel"):
            audio_input = gr.Audio(sources=["microphone"], streaming=True)
        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    bubble_full_width=True, height="65vh", show_copy_button=True
                )
                chat_input = gr.Textbox(
                    interactive=True, placeholder="Type Search Query...."
                )
            with gr.Column(scale=1):
                transcript_textbox = gr.Textbox(
                    lines=40,
                    placeholder="Transcript",
                    max_lines=40,
                    label="Transcript",
                    show_label=True,
                    autoscroll=True,
                )

        chat_input.submit(
            HybridVectorSearch.chat_search, [chat_input, chatbot], [chat_input, chatbot]
        )
        audio_input.stream(
            WhisperAutomaticSpeechRecognizer.transcribe_with_diarization,
            [audio_input, full_stream, transcript],
            [transcript_textbox, full_stream, transcript],
        )

    with gr.Tab("Offline Mode"):
        full_stream = gr.State()
        transcript = gr.State(value="")
        chats = gr.State(value=[])

        with gr.Row(variant="panel"):
            audio_input = gr.Audio(sources=["upload"], type="filepath")
        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    bubble_full_width=True, height="55vh", show_copy_button=True
                )
                chat_input = gr.Textbox(
                    interactive=True, placeholder="Type Search Query...."
                )
            with gr.Column(scale=1):
                transcript_textbox = gr.Textbox(
                    lines=35,
                    placeholder="Transcripts",
                    max_lines=35,
                    label="Transcript",
                    show_label=True,
                    autoscroll=True,
                )

        chat_input.submit(
            HybridVectorSearch.chat_search, [chat_input, chatbot], [chat_input, chatbot]
        )
        audio_input.upload(
            WhisperAutomaticSpeechRecognizer.transcribe_with_diarization_file,
            [audio_input],
            [transcript_textbox, full_stream, transcript],
        )

if __name__ == "__main__":
    demo.launch()
