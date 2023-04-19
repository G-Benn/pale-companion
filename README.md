# Pale Companion

## What is this?
A QA / lookup system ended to help anyone reading Pale by Wildbow (or any other of his serials). 

## How is this Structered?
- `src` contains the source code for building everything. 
    -  `archive` is the complete collection of everything; testing different techniques, implementations, etc.
    -  `processing` contains all the files used to format and process the data.
    -  `GPL` contains the files that helped generate the GPL questions to fine tune the models.
    -  `tune` contains the code used to fine-tune the final model.
    -  `db-gen` contains the files that generate the underlying databases and the pseudo-filtering.
    -  `QA-pale-companion.ipynb` is the final 'production' implementation.

## Technology used
- [HayStack](https://haystack.deepset.ai/) as the backing database system.
- [HuggingFace Transformers](https://huggingface.co/) ([following this excellent book](https://transformersbook.com/))
- [Sentence Transformers](https://www.sbert.net/index.html) as the underlying embeddings
- [Google Colab](https://colab.research.google.com/) as a coding with GPUs resource.

## How can I use it?
### Using it to ask questions
In order to ask it questions you'll need to follow the steps laid out in `QA-pale-companion.ipynb` or the notebook in [Colab](https://colab.research.google.com/drive/12rCqZCSO1N7jzzK6kIszkiUhjN1BKDfn?usp=sharing). In brief this entails downloading the db files from [Google Drive](https://drive.google.com/drive/folders/1CLoVdP7PeEOYi_48QQ0CPLbvrCKcTIH7?usp=share_link), placing them in the same folder as indicated in the notebook, and running it (most likely in Colab). If there are issues, feel free to reach out or put an issue in and I can provide some advice.
### Improve on it / experiment with it
The code is all available here for a reason - please feel free! Please raise issues or ask questions if you encounter any difficulties or have any questions. One comment I'll make is that the underlying serials and their data isn't available per  [author's request](https://www.reddit.com/r/Parahumans/comments/6cusa0/wildbow_ebook_scraper_question/)). If you want to acquire this to experiment with (different embeddings, etc) you'll need to write your scraper for the text.

## Rough process flow
1. Set up data cleaning script to process the HTML of the pages into txt files for lookup and parse out metadata.
    a. Use the very well-maintained [Pale Amalgram sheet](https://docs.google.com/spreadsheets/d/1VS0HRcbHChh4gmL8LcL8xiIvo-nPhSgs2OGOVV3fVbo/edit#gid=0) to match up the metadata rather than generate it entirely myself.
    b. Also make use of the text-based Extra Materials.
    c. Script will flag when there are mismatches between the number of chapter in the spreadsheet and the number of files.
    d. Perform a one-time processing of the completed serials + short stories in `src/processing/process-other-works.ipynb`.
2. Test a simple Extractive QA system for these purposes following [this guide](https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline). Performance wasn't great, but this was a proof-of-concept more than anything.
3. Start experimenting with various other Haystack-recommended systems while I experiment with a few different systems and catch up on my reading and state-of-the-art. Many of these examples are run in Colab for the GPU benefits and are one-off experiments that aren't necessarily well-written or repeatable.
    - Generative v1: RAG following [this](https://haystack.deepset.ai/tutorials/07_rag_generator) in `src/generative_qa_v1`. Worked decent, but isn't expandable.
    - Generative v2: LFQA with a DPR following [this](https://haystack.deepset.ai/tutorials/12_lfqa). The LFQA that powers the generative portion of this and the explanation of the different Generative / Summarization Pipelines is useful and can be expanded on in many ways and this is the general process we followed..
    - Generative Pseudo Labeling (GPL): A method for generating questions from existing documents described here[https://haystack.deepset.ai/tutorials/18_gpl]. This was repeated for each document until we had a set of generated questions for each serial. Some example notebooks can be found in `src/GPL/` and I may publish a few example questions.
    - Fine Tuning: Using the GPL questions above, I iteratively tuned (1 serial at a time since Colab is a short, low-memory environment) the Haystack-recommended [SBERT Ms Marco Distilbert](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) on the QA data. Once this completes this wildbow-tuned model will be pushed to Huggingface where others can potentially look at it.
4. Testing System Development. Based on my reading and experimentation I have a decent idea what the end system here will look like and the testing system here should be similar.
    - Document Store: FAISS for this purpose since it can be easily saved / loaded + run in a Colab environment. However, we do lose the ability to filter on things like chapter numbers.
    - Preprocessor: We'll be using a `max_seq_length=400` for each of the embedding models, iterating on the split length and split_overlap.
    - Retriever: Embedding Retriever so we can work on the SBERT models. This should be the state-of-the-art retriever at the moment and works reasonably well.
    - Embedding Models: Testing 3: The [SBERT Ms Marco Distilbert](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b), [SBERT Ms Marco BERT](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5), and my fine-tuned distilbert (link tbd).
    - End result methodologies:
        - [Extractive Search](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#extractiveqapipeline) as a baseline to find relevant documents and contextual quotes.
        - [Summarization](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#extractiveqapipeline) as a less freeform version of the Generative search.
        - [Generative Seq2Seq](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#generativeqapipeline) model using `vblagoje/bart_lfqa`. I may also experiment with the ELI5 bart (or try my own).
    - Test run location: If I'm able to get a GPU in Colab I may run it there for speed, otherwise local machine should be fine (was not fine).
    - Test run questions: will be present in the notebook and representative of the few arcs + some off-the-wall questions.
    - Time the execution time (in FAISS so not a straight translation to ElasticSearch) to understand the tradeoffs.
    - Learn a reasonable Top K for the Retriever - likely 5-15.
    - Output: Question-Answer pairs keyed on their metadata about the above results.

5. Pull together a comparison to evaluate the above factors. While accuracy is obviously important, since this will need to be run "at scale" keeping in mind timing is important. I'll ingest the QA pairs created from the above methodology and compare them in a few key area: execution speed and "quality". I'll rank the outputs and may offer an "Option" of 2 different end results since something Generative vs Summarative may be helpful for different use cases.

6. Create an implementation for Pale that is reasonably stand-alone. This won't require someone to interface with something like Colab or other compute-intensive steps, instead just loading a pre-computed embedding store and asking questions.
    - The end result here is a notebook in Colab that runs through everything + a set of series-centered pre-computed FAISS databases.
    - Other ideas explored included:
        - This will most likely be deployed via Streamlit since it can be containerized reasonably easily. Some of this will depend on how ElasticSearch works.
        - This will most likely be handled by a [REST API](https://haystack.deepset.ai/tutorials/20_using_haystack_with_rest_api) implementation of Haystack.
        - Document Store: ElasticSearch in production (save the container = save the embeddings). I'll likely need to create this iteratively and save often as I'm not sure if it'll be able to be create all serials at once (though I'll start with just Pale). 
    - Embedding model: The finetuned [wildbow-distilbert](https://huggingface.co/TheSpaceManG/wildbow-distilbert), though others can be used.
    - Top K for Retriever: Toggled by the user (with above-informed default of 15).
    - As part of the retriever output, include metadata information regardless of the methodology. Information about the chapters, the viewpoints that generate the information, etc may be informative a user.

7. This would only be possible if I was able to get ElasticSearch or similar Database functioning, which wasn't possible on my current machine or Colab. This is a limitation of the tech, but if I ever upgrade my old GPU we'll see if I revisit this. ~~In the user application, implement metadata filtering. This will help to inform spoiler-free usage. It'll require a user-entered chapter number -> series chapter number (since the Lexigraphical order can't be used here) mapping function but should be helpful. While the initial deployment will only work on Pale, the end goal is something that can toggle between Serials.~~

8. In theory this isn't hard (just recomputing the embeddings for the new chapters and adding them to the Document Store), but it's not worth the current time. ~~Develop a plan / process for continual updates of chapters.~~

9. In progress! ~~Expose this to the broader internet. Publicize it through Reddit or Discord. It may start out as a "Hey download this and run it  yourself with X" or something bigger - yet to be seen.~~

10. The GPU-intensiveness of the generation right now would likely blow through any free credits here so this is on pause - consider the Colab notebook this. ~~Get it up into AWS / GCP / Azure / RPi - anything to make it a standalone service that doesn't cost much.~~

11. Possible future work:
    - Work in developing my own [custom model input converters](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/answer_generator/transformers.py#L482) that can utilize other text2text models from [Huggingface](https://huggingface.co/models?pipeline_tag=text2text-generation).
    - Look into the possibility of biasing the Retrievers to preferring more recent results - knowing who A was in chaper 10 is more beneficial than chapter 1.
    - For illustration purposes build a [Question Generation system](https://docs.haystack.deepset.ai/docs/ready_made_pipelines#questiongenerationpipeline)
    - Can we make use of fan speculation (Reddit, Discord, Spacebattles etc), Word-of-God, or other sources (Pale Reflections, etc) to augment these responses?
    - Can I make it read back responses for Accessibility purposes?
    - Change of database to allow filtering on metadata (eg only Pale, only chapters < 10.1, only Avery PoV, etc)
    - Different styles of answer generation. This type of work is relatively simple to do within the Haystack framework but it relies on a custom [Converter](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/answer_generator/transformers.py#L481) being written. See [this](https://yjernite.github.io/lfqa.html) for an explanation on what is involved in getting a new LFQA generator working.
    - A Generator that can better mimic the style of Wildbow's answers. This would rely on a source-of-truth set of Document:Summary data that would be difficult to compile. However, it would both provide a more familiar output style and potentially could teach the model some of the vocabulary quirks that are present in the worlds (eg Other in the PactVerse != other in common usage). Similar to the above.
    - A better way to capture rare words (like Primordial, Aurum) in the answers - in NLP this is called [temperature](https://lukesalamone.github.io/posts/what-is-temperature/). While you typically don't want high amounts of creativity in your QA system, it can help make up for other shortcomings like the lack of trained vocabulary. This would require [MonkeyPatching](https://stackoverflow.com/questions/5626193/what-is-monkey-patching) or other general modifications of the [Haystack source code](https://github.com/deepset-ai/haystack/blob/main/haystack/nodes/answer_generator/transformers.py#L465) which could be a large endeavor. 
    - A better pretraining on real questions and answers. The pretraining here is done via a technique called [Generative Pseudolabeling](https://www.pinecone.io/learn/gpl/), but it's no substitute for stronger training data for domain adaption.
  - The inclusion of various Word-of-God quotes or user-submitted answers to other queries (ie Reddit, Discord). The difficulty here is parsing the data formats and gathering it into formats that 1) better inform the Document embeddings how to store the information, 2) can be searched efficiently, and 3) are correct.
    - Updating Pale chapters available - currently the initial dataset only includes Pale chapters up to 23.1. This would be an incremental update.
    - Incorporating the possibility of a MultiModal Transformer for the Extra Materials. Currently any information from the Pale Extra Materials are included via transcripts manually pulled from the site(many thanks to those who transcribed this information) but the possibility of searching text and image embeddings simultaneously is a possibility.
   - Not really related to this but an idea - "A boar by any other name" ie what do people refer to Wildbow as to avoid a bing (from the IRC days)
