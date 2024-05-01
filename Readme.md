# Live Speech Reference Search

## Problem
I faced the task of enabling search within meeting transcripts, spanning discussions among 2 to 4 participants, whether conducted online or offline. The aim was to seamlessly reference past discussions during live conversations within the same group.

## Objective
The objective was to develop a foundational solution that could later evolve to address this challenge comprehensively. It served as a checkpoint to identify all essential considerations for tackling the aforementioned problem, including:
- Real-time transcription of live conversations, regardless of their online or offline nature.
- Retrieval of past meeting conversations within the same group.
- Ensuring the accuracy and efficiency of reference searches.
- Exploring various scenarios to be accounted for in the solution design process.

## Demo Video



https://github.com/SubtleParesh/live-speech-reference-search/assets/20494680/fc72e398-cd21-4001-a0c6-d42883d36331

https://github.com/SubtleParesh/live-speech-reference-search/assets/20494680/8c50b654-ca3a-495a-86fd-7b88ef160073



## [Blog Post](https://medium.com/@subtleparesh/journey-into-automatic-speech-recognition-build-your-own-real-time-searchable-transcript-system-1929eb351593) 

# Running the project

### Update Hugging face token
Update environment variable for hugging face token in `environment.yml`

### Creating & Activating Environment from conda
`conda env create -f environment.yml`

`conda activate live_speech_reference_search`

### Start Qdrant
`docker compose up`

### Start UI
`gradio app.py`

### UI Accessible at http://localhost:7860/


## Troubleshooting

#### Issue with spacy - Run following command
`python -m spacy download en_core_web_sm`

####  Ffmpeg must be installed

#### This is tested on GPU with Memory 6gb, You can change device to run on CPU for getting it running


# References
- https://pytorch.org/audio/stable/_modules/torchaudio/datasets/tedlium.html
- https://qdrant.tech/articles/sparse-vectors/
- https://www.pinecone.io/learn/splade/
- https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
