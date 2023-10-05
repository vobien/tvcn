import streamlit as st

from common import load_data, load_model, load_tokenized_data, search, download_files, ranking

@st.cache_resource()
def download_data():
    download_files()

@st.cache_resource()
def load_model_visbert_and_data():
    # model keepitreal/vietnamese-sbert
    passages = load_data()
    vi_sbert, vi_sbert_corpus_embeddings = load_model(passages, model_name='keepitreal/vietnamese-sbert')
    return passages, vi_sbert, vi_sbert_corpus_embeddings


@st.cache_resource()
def load_model_simcse_and_data():
    # model SimCSE & tokenize
    tokenized_passages = load_tokenized_data()
    sim_sce, sim_sce_corpus_embeddings = load_model(tokenized_passages, model_name='VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    return sim_sce, sim_sce_corpus_embeddings


def run(top_k=10):
    st.title('Tìm kiếm theo ngữ nghĩa')
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', ["vietnamese-sbert", "SimCSE"], index=0)
    st.markdown('Các bạn có thể gõ câu hỏi tiếng việt có dấu nào ở trong ô bên dưới và bấm nút search để tra cứu sách của trưởng lão Thích Thông Lạc.')
    st.text('')
    input_text = []
    comment = st.text_area('Vui lòng nhập câu hỏi tiếng việt có dấu ở đây!')
    input_text.append(comment)

    if st.button('Tìm kiếm'):
        with st.spinner('Searching ......'):
            if input_text != '':
                print(f'INPUT: ', input_text)
                query = input_text[0]
                if ranker == 'vietnamese-sbert':
                    results, hits = search(vi_sbert, vi_sbert_corpus_embeddings, query, passages, top_k=top_k)
                    # results, hits = ranking(hits, query, passages)
                elif ranker == 'SimCSE':
                    results, hits = search(sim_sce, sim_sce_corpus_embeddings, query, passages, is_tokenize=True, top_k=top_k)
                    # results, hits = ranking(hits, query, passages)

                for result in results:
                    st.success(f"{str(result)}")


if __name__ == '__main__':
    download_data()
    passages, vi_sbert, vi_sbert_corpus_embeddings = load_model_visbert_and_data()
    sim_sce, sim_sce_corpus_embeddings = load_model_simcse_and_data()
    run(top_k=100)