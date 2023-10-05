import streamlit as st

from common import load_data, load_model, load_tokenized_data, search, download_files, ranking
from bm25 import lexical_search, load_bm25

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
def load_model_bm25():
    return load_bm25(passages)


def run(model_names, model_mapping, top_k=10):
    st.title('Tìm kiếm theo ngữ nghĩa')
    rankers = model_names[:]
    # rankers.append("lexical-search-bm25")
    rankers.append("ensemble")
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', rankers, index=0)
    st.markdown("Các bạn có thể nhập thắc mắc của mình bằng tiếng việt có dấu ở ô bên dưới và bấm nút search để tra cứu sách của trưởng lão Thích Thông Lạc." 
     + "Kết quả tra cứu chính xác 100% là kinh sách của Trưởng Lão Thích Thông Lạc."
     + "Vì trang web chỉ tìm kiếm thông tin gần đúng với những gì bạn nhập vào, cho nên bạn nên đọc qua các kết quả tìm được để chọn thông tin phù hợp nhất với bạn."
     + "Bạn có thể lựa chọn các mô hình ngôn ngữ khác nhau để đa dạng kết quả tìm được."
     + "Bạn cũng có thể đánh dấu chọn vào ô 'Ranking kết quả tìm kiếm' để thuật toán sắp xếp lại thứ tự kết quả phù hợp nhất.")
    st.text('')
    input_text = []
    comment = st.text_area('Vui lòng nhập câu hỏi tiếng việt có dấu ở đây!')
    input_text.append(comment)

    is_ranking = st.checkbox("Ranking lại kết quả tìm kiếm")

    if st.button('Tìm kiếm'):
        with st.spinner('Searching ......'):
            if input_text != '':
                print(f'Input: ', input_text)
                query = input_text[0]
                if ranker == 'lexical-search-bm25':
                    results, _  = lexical_search(bm25, query, passages, top_k=top_k)
                elif ranker == "ensemble":

                    ids = []
                    for model_name in model_names:
                        model = model_mapping[model_name]["model"]
                        corpus = model_mapping[model_name]["corpus"]
                        results, hits = search(model, corpus, query, passages, top_k=top_k)
                        
                        for hit in hits:
                            ids.append(hit["corpus_id"])
                    
                    print(ids)
                    ids = set(ids)
                    print(ids)

                    hits = [{"corpus_id": id} for id in ids]
                    results = [passages[i] for i in ids]

                    if is_ranking:
                        results, _ = ranking(hits, query, passages, top_k=top_k)
                else:
                    print("Search answers with model ", ranker)
                    model = model_mapping[ranker]["model"]
                    corpus = model_mapping[ranker]["corpus"]
                    results, hits = search(model, corpus, query, passages, top_k=top_k)
                    
                    if is_ranking:
                        results, _ = ranking(hits, query, passages, top_k=top_k)

                for result in results:
                    st.success(f"{str(result)}")


if __name__ == '__main__':
    model_names = [
        "keepitreal/vietnamese-sbert",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]

    download_data()
    passages = load_input_data()
    model_mapping = load_model_and_corpus(model_names)
    # bm25 = load_model_bm25()
    run(model_names, model_mapping, top_k=10)