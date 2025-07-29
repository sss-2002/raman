import streamlit as st

def main():

  # 返回主页按钮
    if st.button("返回主页"):
       st.session_state.current_page = 'home'
       st.experimental_rerun()
    st.title("西安电子科技大学生物光学实验室（BIOLIGHT LAB）介绍")
    st.markdown("西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月。")
    st.markdown("作为智能医学检测技术的创造者和实践者，实验室始终秉持着成长型思维，全力打造一支勇往直前的生物态团队。")
    st.markdown("实验室的核心目标是培养富有创新精神和实践能力的新时代人才，为生物光学领域的发展注入新的活力。")
    st.markdown("在科研方面，实验室专注于生物光学技术在医学检测中的应用，致力于开发更加精准、高效的检测方法和设备。")
    st.markdown("同时，实验室也积极开展国际合作与交流，与国内外知名科研机构和企业保持着密切的联系。")

   
