import streamlit as st
from modules.chunk.runChunk import process_and_store_documents

st.title("📂 Tải lên và xử lý tài liệu")

uploaded_files = st.file_uploader("Tải lên tài liệu", accept_multiple_files=True)

if uploaded_files and st.button("🔄 Xử lý tài liệu"):
    if "client" not in st.session_state:
        st.warning("Vui lòng khởi tạo mô hình và cơ sở dữ liệu từ trang chính.")
    else:
        for uploaded_file in uploaded_files:
            process_and_store_documents(uploaded_file, st.session_state["client"])
        st.success("Tất cả tài liệu đã được xử lý và lưu trữ.")
