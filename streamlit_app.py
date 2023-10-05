import streamlit as st

from common import load_data, load_model, load_tokenized_data, search, download_files, ranking

@st.cache_resource()
def download_data():
    download_files()

@st.cache_resource()
def load_input_data():
    return load_data()

@st.cache_resource()
def load_model_and_corpus(model_names):
    model_mapping = {}
    for model_name in model_names:
        model, corpus = load_model(passages, model_name=model_name)
        model_mapping[model_name] = {
            "model": model,
            "corpus": corpus
        }
    return model_mapping

@st.cache_resource()
def load_model_simcse_and_data():
    # model SimCSE & tokenize
    tokenized_passages = load_tokenized_data()
    sim_sce, sim_sce_corpus_embeddings = load_model(tokenized_passages, model_name='VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    return sim_sce, sim_sce_corpus_embeddings


def run(model_names, model_mapping, top_k=10):
    st.title('Tìm kiếm theo ngữ nghĩa')
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', model_names, index=0)
    st.markdown('Các bạn có thể gõ câu hỏi tiếng việt có dấu nào ở trong ô bên dưới và bấm nút search để tra cứu sách của trưởng lão Thích Thông Lạc.')
    st.text('')
    input_text = []
    comment = st.text_area('Vui lòng nhập câu hỏi tiếng việt có dấu ở đây!')
    input_text.append(comment)

    if st.button('Tìm kiếm'):
        with st.spinner('Searching ......'):
            if input_text != '':
                print(f'Input: ', input_text)
                query = input_text[0]
                if ranker != '':
                    print("Search answers with model ", ranker)
                    model = model_mapping[ranker]["model"]
                    corpus = model_mapping[ranker]["corpus"]
                    results, _ = search(model, corpus, query, passages, top_k=top_k)

                for result in results:
                    st.success(f"{str(result)}")


if __name__ == '__main__':
    model_names = [
        "keepitreal/vietnamese-sbert",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]

    # download_data()
    passages = load_input_data()
    model_mapping = load_model_and_corpus(model_names)
    run(model_names, model_mapping, top_k=100)