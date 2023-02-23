# Pale Companion

## What is this?
A QA / lookup system ended to help anyone reading Pale by Wildbow (and hopefully also the other series in the future). 

## How is this Structered?
- `data` contains the training data for fine-tuning the models and building the retreival systems. There's a folder of the HTML files of the chapters (`pale`) themselves and a folder of the transcripts corresponding to the extra materials (`pale-extra-materials`) with a few already-processed Pickle files that the model(s)
- `src` contains the source code for building everything (still in progress). Generally the Jupyter notebooks are for initial testing and experimentation while .py scripts are for finalized code.

## Technology used
- [HayStack](https://haystack.deepset.ai/) 
- [HuggingFace Transformers](https://huggingface.co/) ([following this excellent book](https://transformersbook.com/))
- [Sentence Transformers](https://www.sbert.net/index.html)
- StreamLit

## How can I use it?
As a standalone product? TBD. The plan on my end is to deploy this as a service somewhere in the free tier of AWS and you can interact with the application. As some examples or to test for yourself? Install all the packages in `requirements.txt` and give it a spin! If you don't have a GPU and/or a linux-based system there may be some additional adaption needed as this was built / tested in a WSL2 environment with a weak CUDA-enabled GPU. I plan to version a GPL-informed fine-tuned version to Huggingface and release all other information (short of the files themselves per the [author's request](https://www.reddit.com/r/Parahumans/comments/6cusa0/wildbow_ebook_scraper_question/))

## Current plans
1. Set up data cleaning script to process the HTML of the pages into txt files for lookup and parse out metadata. <-- Currently converting this from a notebook (`src/format-files.ipynb`) to a script
    a. Use the very well-maintained [Pale Amalgram sheet](https://docs.google.com/spreadsheets/d/1VS0HRcbHChh4gmL8LcL8xiIvo-nPhSgs2OGOVV3fVbo/edit#gid=0) to match up the metadata rather than generate it entirely myself.
    b. Also make use of the text-based Extra Materials.
    c. Script will flag when there are mismatches between the number of chapter in the spreadsheet and the number of files.
    d. Perform a one-time processing of the completed serials + short stories in `src/process-other-works.ipynb`.
2. Test a simple Extractive QA system for these purposes following [this guide](https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline). Performance wasn't great, but this was a proof-of-concept more than anything. See `src/extractive_qa.ipynb` for this example.
3. Start experimenting with various other Haystack-recommended systems while I experiment with a few different systems and catch up on my reading and state-of-the-art. Many of these examples are run in Colab for the GPU benefits and are one-off experiments that aren't necessarily well-written or repeatable.
    - Generative v1: RAG following [this](https://haystack.deepset.ai/tutorials/07_rag_generator) in `src/generative_qa_v1`. Worked decent, but isn't expandable.
    - Generative v2: LFQA with a DPR following [this](https://haystack.deepset.ai/tutorials/12_lfqa). Works decent, but DPR isn't simple to use or improve on and has token limitations. However, the LFQA that powers the generative portion of this and the explanation of the different Generative / Summarization Pipelines is useful.
    - Generative Pseudo Labeling (GPL): A method for generating questions from existing documents described here[https://haystack.deepset.ai/tutorials/18_gpl]. This was repeated for each document until we had a set of generated questions for each serial. Some example notebooks can be found in `src/GPL/` and I may publish a few example questions.
    - Fine Tuning: Using the GPL questions above, I iteratively tuned (1 serial at a time since Colab is a short, low-memory environment) the Haystack-recommended [SBERT Ms Marco Distilbert](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) on the QA data. Once this completes this wildbow-tuned model will be pushed to Huggingface where others can potentially look at it.
4. Testing System Development. Based on my reading and experimentation I have a decent idea what the end system here will look like and the testing system here should be similar.
    - Document Store: FAISS for this purpose, with the first few chapters / arcs (through 5-10) for speed purposes
    - Preprocessor: We'll be using a `max_seq_length=400` for each of the embedding models, iterating on the split length and split_overlap
    - Retriever: Embedding Retriever so we can work on the SBERT models. This should be the state-of-the-art retriever at the moment and works reasonably well.
    - Embedding Models: Testing 3: The [SBERT Ms Marco Distilbert](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b), [SBERT Ms Marco BERT](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5), and my fine-tuned distilbert (link tbd).
    - End result methodologies:
        - [Extractive Search](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#extractiveqapipeline) as a baseline to find relevant documents and contextual quotes.
        - [Summarization](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#extractiveqapipeline) as a less freeform version of the Generative search.
        - [Generative Seq2Seq](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#generativeqapipeline) model using `vblagoje/bart_lfqa`. I may also experiment with the ELI5 bart (or try my own).
    - Test run location: If I'm able to get a GPU in Colab I may run it there for speed, otherwise local machine should be fine.
    - Test run questions: will be present in the notebook and representative of the few arcs + some off-the-wall questions.
    - Time the execution time (in FAISS so not a straight translation to ElasticSearch) to understand the tradeoffs.
    - Learn a reasonable Top K for the Retriever - likely 5-15.
    - Output: Question-Answer pairs keyed on their metadata about the above results.

5. Pull together a comparison to evaluate the above factors. While accuracy is obviously important, since this will need to be run "at scale" keeping in mind timing is important. I'll ingest the QA pairs created from the above methodology and compare them in a few key area: execution speed and "quality". I'll rank the outputs and may offer an "Option" of 2 different end results since something Generative vs Summarative may be helpful for different use cases.

6. Create an implementation for Pale that is reasonably stand-alone. This won't require someone to interface with something like Colab or other compute-intensive steps, instead just loading a pre-computed embedding store and asking questions.
    - This will most likely be deployed via Streamlit since it can be containerized reasonably easily. Some of this will depend on how ElasticSearch works.
    - This will most likely be handled by a [REST API](https://haystack.deepset.ai/tutorials/20_using_haystack_with_rest_api) implementation of Haystack.
    - Document Store: ElasticSearch in production (save the container = save the embeddings). I'll likely need to create this iteratively and save often as I'm not sure if it'll be able to be create all serials at once (though I'll start with just Pale). 
    - Embedding model: TBD
    - Methodology: TBD, potentially toggled
    - Top K for Retriever: Toggled by the user (with above-informed default + guardrails)
    - As part of the retriever output, include metadata information regardless of the methodology. Information about the chapters, the viewpoints that generate the information, etc may be informative a user.

7. In the user application, implement metadata filtering. This will help to inform spoiler-free usage. It'll require a user-entered chapter number -> series chapter number (since the Lexigraphical order can't be used here) mapping function but should be helpful. While the initial deployment will only work on Pale, the end goal is something that can toggle between Serials.

8. Develop a plan / process for continual updates of chapters. My hope is that the usage of ElasticSearch can allow me to add new chapters as they release (and are cleaned) and only update embeddings on those chapters.

9. Expose this to the broader internet. Publicize it through Reddit or Discord. It may start out as a "Hey download this and run it  yourself with X" or something bigger - yet to be seen.

10. Get it up into AWS / GCP / Azure / RPi - anything to make it a standalone service that doesn't cost much.

11. Possible future work:
    - More work to allow user flexibility in the responses
    - Work in developing my own [custom model input converters](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/answer_generator/transformers.py#L482) that can utilize other text2text models from [Huggingface](https://huggingface.co/models?pipeline_tag=text2text-generation).
    - Look into the possibility of biasing the Retrievers to preferring more recent results - knowing who A was in chaper 10 is more beneficial than chapter 1.
    - COntinue improving on efficiency with ONNX or other technology.
    - Write better guides on getting this running.
    - Expand this out to other Wildbow Serials
    - Not really related to this but an idea - "A boar by any other name" ie what do people refer to Wildbow as to avoid a bing (from the IRC days)
    - For illustration purposes build a [Question Generation system](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#questiongenerationpipeline)
    - Incorporate metadata about chapter perspectives into the generated responses instead of an "oh by the way this response was from XYZ perspective(s)"
    - Can we make use of fan speculation (Reddit, Discord, Spacebattles etc), Word-of-God, or other sources (Pale Reflections, etc) to augment these responses?
    - Can we do any image-embeddings to augment the text transcripts of the Extra Materials? We currently have user/author-provided transcripts but MultiModal transformers do exist.
    - Can I make it read back responses for Accessibility purposes?
