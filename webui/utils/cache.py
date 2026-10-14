import streamlit as st
import os

def get_fonts_cache(font_dir):
    if 'fonts_cache' not in st.session_state:
        fonts = []
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                # 支持常见字体格式，少字体时也能被UI识别
                if file.lower().endswith((".ttf", ".ttc", ".otf")):
                    fonts.append(file)
        fonts.sort()
        st.session_state['fonts_cache'] = fonts
    return st.session_state['fonts_cache']

