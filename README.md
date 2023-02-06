# Pale Companion

## What is this?
A QA / lookup system ended to help anyone reading Pale by Wildbow (and hopefully also the other series in the future). 

## How is this Structered?
- `data` contains the training data for fine-tuning the models and building the retreival systems. There's a folder of the HTML files of the chapters (`pale`) themselves and a folder of the transcripts corresponding to the extra materials (`pale-extra-materials`) with a few already-processed Pickle files that the model(s)
- `src` contains the source code for building everything (still in progress). Generally the Jupyter notebooks are for initial testing and experimentation while .py scripts are for finalized code.

## Technology used
- HayStack 
- HuggingFace Transformers
- StreamLit
- CUDA GPU
## How can I use it?
As a standalone product? TBD. As some examples or to test for yourself? Install all the packages in `requirements.txt` and give it a spin! If you don't have a GPU and/or a linux-based system there may be some additional adaption needed as this was built / tested in a WSL2 environment with a CUDA-enabled GPU.

## Steps and ongoing progress
1. Set up data cleaning script to process the HTML of the pages into txt files for lookup and parse out metadata. <-- Currently converting this from a notebook to a script
    a. Use the very well-maintained [Pale Amalgram sheet](https://docs.google.com/spreadsheets/d/1VS0HRcbHChh4gmL8LcL8xiIvo-nPhSgs2OGOVV3fVbo/edit#gid=0) to match up the metadata rather than generate it entirely myself.
    b. Also make use of the text-based Extra Materials.
    c. Script will flag when there are mismatches between the number of chapter in the spreadsheet and the number of files.
2. Spin up initial simple QA systems:
    - Extractive: following [this guide](https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline) <-- This works decent
    - Generative v1: following [this](https://haystack.deepset.ai/tutorials/07_rag_generator) <-- Being worked on
    - Generative v2: following [this](https://haystack.deepset.ai/tutorials/12_lfqa)
    - Read through and implement the [recommended Preprocessing steps]() (may be different for different models to input)
3. Experiment with the success in using a couple different BERT models. Some factors to consider: We want to run this on an inexpensive machine, so we shold try a small model. We want to ensure that it runs reasonably quick.
4. Fine-tune a model on both the chapters themselves as well as Other Wildbow works to see if it improves performance. Will most likely use the top models from step 3) here and train them in Google Colab for the GPU benefits utilizing HuggingFace Transformers. This should be done in such a manner that I can fine-tune it on individual chapters as they come in.
5. Put this on top of a StreamLit application (or other WebApp solution) to expose a basic UI and toggle between different choices of models, top-k, Retreivers, etc for internal testing purposes. This would 
6. Upgrade the QA system (and Streamlit app) to use Metadata to limit the chapter searches. This would allow it to be used spoiler-free by people who aren't up-to-date with the current chapters. Can we also bias the answers / retrieval to skew towards more recent chapters?
7. Come up with a efficient method for continual learning and adding in new chapter incrementally as they release without going through the expensive steps of training from Scratch.
8. Expand the simple QA pipeline with a more efficient one build on ElasticCache following [this guide](https://haystack.deepset.ai/tutorials/03_scalable_qa_system)
9. Experiment with different retrieval implementations [ex](https://haystack.deepset.ai/tutorials/06_better_retrieval_via_embedding_retrieval)
10. Try implementing a REST version of the pipeline that may be easier for online implementation. [ex](https://haystack.deepset.ai/tutorials/20_using_haystack_with_rest_api)
11. Package it up so that it can be deployed within a free version of AWS, GCP, or Azure and exposed to the world. This may also be done with a Raspberry Pi or other affordable hosting solution. Conversion to ONNX or other hyper-efficient format may be a solution here.
12. Expand out the QA system to other Series.
13. Expand a summarization system.
14. Expand on the returns with various nice-to-haves and thoughts as we go along
    - Might be informative to mention _who_ is giving us the answer as that may change our perpsective on it.
    - Can we make use of fan speculation (Reddit, Discord, etc), Word-of-God, or other sources (Pale Reflections) to augment these responses?
    - Can we do any image-embeddings to augment the text transcripts of the Extra Materials?
    - Can I make it talk for Accessibility purposes?