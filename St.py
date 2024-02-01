'''
    运用Streamlit完成前端页面显示
'''
from Rag import Rag
import streamlit as st

with st.sidebar:
    job_id = st.text_input("工号", key="job_id")
    name = st.text_input("姓名", key="name")
    tips = st.checkbox("温馨提示:您与大模型的对话都会被记录,并有可能被用于训练该大模型,勾选即表示您知情并同意",
                       value=False, key=None)
st.title("💬 RAG大模型")
st.caption("🚀 聊天机器人developed by IT")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，有什么可以帮您?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "您好，有什么可以帮您?"}]
st.sidebar.button('清空聊天记录', on_click=clear_chat_history)


if prompt := st.chat_input():
    if not job_id:
        st.info("请输入工号.")
        st.stop()
    if not name:
        st.info("请输入姓名.")
        st.stop()
    if not tips:
        st.info("请先选中知情同意书.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # 调用RAG类
    msg = Rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
