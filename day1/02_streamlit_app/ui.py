# ui.py
import streamlit as st
import pandas as pd
import html
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions
import datetime

# --- LINE風スタイルの定義 ---
def apply_line_style():
    """LINE風のスタイルをアプリに適用する"""
    # LINEのブランドカラー
    line_primary_color = "#06C755"  # LINE緑
    line_background = "#FFFFFF"  # 白背景
    line_chat_bg = "#F5F5F5"  # チャット画面背景色（薄いグレー）
    line_text_color = "#333333"  # テキスト色
    line_light_gray = "#DDDDDD"  # 薄いグレー（区切り線など）
    line_timestamp = "#AAAAAA"  # タイムスタンプの色
    
    # CSSスタイルをapply
    st.markdown(f"""
    <style>
        /* 全体の背景とテキスト */
        .stApp {{
            background-color: {line_chat_bg};
            color: {line_text_color};
            font-family: 'Hiragino Kaku Gothic Pro', 'ヒラギノ角ゴ Pro W3', Meiryo, 'メイリオ', sans-serif;
        }}
        
        /* ヘッダースタイル */
        .stTitleContainer h1 {{
            color: {line_primary_color} !important;
            font-weight: bold;
        }}
        
        /* ボタンスタイル */
        .stButton>button {{
            background-color: {line_primary_color};
            color: white;
            border-radius: 6px;
            font-weight: bold;
            border: none;
            padding: 0.3rem 1rem;
        }}
        
        /* テキストエリアスタイル */
        .stTextArea textarea {{
            border-radius: 20px;
            border: 1px solid #999999;  /* 境界線を濃くする */
            padding: 10px;
            background-color: #FFFFFF;  /* 背景色を白に設定 */
            color: #333333;  /* テキスト色を濃く設定 */
            font-weight: normal;  /* テキストを標準の太さに */
            caret-color: {line_primary_color} !important;
        }}

        /* テキストエリアのラベル */
        .stTextArea label {{
            color: #333333 !important;  /* ラベルの色を濃く設定 */
            font-weight: bold;  /* ラベルを太字に */
        }}
        
        /* セレクトボックスのテキスト */
        .stSelectbox label, .stSelectbox div[data-baseweb="select"] {{
            color: #333333 !important;
        }}

        /* LINE風チャット画面のコンテナ */
        .chat-container {{
            background-color: {line_chat_bg};
            padding: 10px;
            overflow-y: auto;
            border-radius: 5px;
        }}
        
        /* 自分（右側）のメッセージ */
        .user-message {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 15px;
        }}
        
        .user-bubble {{
            background-color: {line_primary_color};
            color: white;
            border-radius: 20px;
            padding: 10px 15px;
            max-width: 70%;
            position: relative;
            margin-right: 10px;
            word-wrap: break-word;
        }}
        
        /* 相手（左側）のメッセージ */
        .bot-message {{
            display: flex;
            justify-content: flex-start;
            margin-bottom: 15px;
        }}
        
        .bot-avatar {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: white;
            margin-right: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            border: 1px solid {line_light_gray};
            overflow: hidden;
        }}
        
        .bot-bubble {{
            background-color: white;
            color: {line_text_color};
            border-radius: 20px;
            padding: 10px 15px;
            max-width: 70%;
            position: relative;
            word-wrap: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        
        /* タイムスタンプ */
        .timestamp {{
            font-size: 11px;
            color: {line_timestamp};
            margin-top: 5px;
            text-align: right;
        }}
        
        /* タブのスタイル */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
            font-weight: bold;
            font-size: 14px;
            color: #333333 !important;
        }}
        
        /* フィードバックエリア */
        .feedback-area {{
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid {line_light_gray};
        }}
        
        /* LINE風区切り線 */
        .line-divider {{
            text-align: center;
            margin: 15px 0;
            color: {line_timestamp};
            font-size: 12px;
            position: relative;
        }}
        
        .line-divider:before, .line-divider:after {{
            content: "";
            display: inline-block;
            width: 40%;
            height: 1px;
            background: {line_light_gray};
            margin: 0 10px;
            vertical-align: middle;
        }}

        /* メトリクス表示のスタイル */
        .stMetric {{
            color: {line_text_color} !important;
        }}
        
        .stMetric .st-emotion-cache-1wivap2 {{  /* メトリクス値の色を濃くする */
            color: {line_text_color} !important;
        }}
        
        .stMetric .st-emotion-cache-r421ms {{  /* メトリクスラベルの色を濃くする */
            color: {line_text_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- LINE風メッセージ表示用ヘルパー関数 ---
def display_user_message(content, timestamp=None):
    """ユーザーのメッセージを右側に表示"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%H:%M")
    
    st.markdown(f"""
    <div class="user-message">
        <div>
            <div class="user-bubble">{content}</div>
            <div class="timestamp">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_bot_message(content, timestamp=None):
    """ボットのメッセージを左側に表示"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%H:%M")
    
    # 空白や空文字列の場合は表示しない
    if content is None or content.strip() == "":
        content = "メッセージを表示できません。"
    
    # HTMLタグをエスケープして安全に表示
    safe_content = html.escape(content)
    # 改行をHTMLの改行タグに変換
    safe_content = safe_content.replace('\n', '<br>')
    
    st.markdown(f"""
    <div class="bot-message">
        <div class="bot-avatar">AI</div>
        <div>
            <div class="bot-bubble">{safe_content}</div>
            <div class="timestamp">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_date_divider(date_str):
    """日付の区切り線を表示"""
    st.markdown(f"""
    <div class="line-divider">{date_str}</div>
    """, unsafe_allow_html=True)

# --- チャットページのUI ---
def display_chat_page(pipe):
    """チャットページのUIを表示する"""
    # LINE風スタイルを適用
    apply_line_style()
    
    # セッション状態の初期化
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # タイトルとサブタイトル
    st.subheader("📱 AIアシスタント")
    
    # チャット履歴の表示
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 現在の日付を表示
        today = datetime.datetime.now().strftime("%Y年%m月%d日")
        display_date_divider(today)
        
        # チャット履歴を表示
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                display_user_message(msg["content"], msg["timestamp"])
            else:
                display_bot_message(msg["content"], msg["timestamp"])
        
        # 現在の会話を表示（フィードバック前のみ）
        if st.session_state.current_question and not st.session_state.feedback_given:
            display_user_message(st.session_state.current_question)
            
            # ボットの応答を表示
            if st.session_state.current_answer:
                display_bot_message(st.session_state.current_answer)
                
                # フィードバックフォームを表示
                display_feedback_form()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 入力エリア（LINE風の入力ボックス）
    st.markdown("<br><br>", unsafe_allow_html=True)  # スペースを空ける
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_area("メッセージを入力", key="question_input", height=70, 
                                    value="")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # 位置調整
        submit_button = st.button("送信")
    
    # 送信ボタンが押された場合
    if submit_button and user_question:
        st.session_state.current_question = user_question
        st.session_state.current_answer = ""
        st.session_state.feedback_given = False
        
        with st.spinner("入力中..."):
            try:
                answer, response_time = generate_response(pipe, user_question)
                
                # デバッグ情報（問題診断用）
                if answer is None or answer.strip() == "":
                    st.error("LLMからの応答が空です。llm.pyのgenerate_response関数を確認してください。")
                    answer = "応答を取得できませんでした。"
                    
                st.session_state.current_answer = answer
                st.session_state.response_time = response_time
                
                # 明示的に再描画
                st.rerun()
            except Exception as e:
                import traceback
                st.error(f"エラーが発生しました: {str(e)}")
                st.code(traceback.format_exc())
                st.session_state.current_answer = f"エラーが発生しました: {str(e)}"

def display_feedback_form():
    """LINE風のフィードバックスタンプを表示"""
    st.markdown("<div class='feedback-area'>", unsafe_allow_html=True)
    st.markdown("### この回答はどうでしたか？")
    
    # フォーム内容
    with st.form("feedback_form"):
        cols = st.columns(3)
        with cols[0]:
            like = st.checkbox("👍 正確!", key="like_checkbox")
        with cols[1]:
            neutral = st.checkbox("🤔 まあまあ", key="neutral_checkbox")
        with cols[2]:
            dislike = st.checkbox("👎 いまいち", key="dislike_checkbox")
        
        # チェックボックスからフィードバック値を計算
        feedback = "正確" if like else ("部分的に正確" if neutral else "不正確")
        
        # フィードバックテキスト入力（省略可能）
        correct_answer = st.text_area("より良い回答の提案（省略可能）", key="correct_answer_input", height=80, 
                                     help="より適切な回答がある場合に入力してください")
        feedback_comment = st.text_area("コメント（省略可能）", key="feedback_comment_input", height=80,
                                      help="その他のフィードバックを入力してください")
        
        # 送信ボタン
        cols = st.columns([3, 1])
        with cols[1]:
            submitted = st.form_submit_button("送信")
        
        if submitted:
            # フィードバックをデータベースに保存
            is_correct = 1.0 if feedback == "正確" else (0.5 if feedback == "部分的に正確" else 0.0)
            combined_feedback = f"{feedback}"
            if feedback_comment:
                combined_feedback += f": {feedback_comment}"

            save_to_db(
                st.session_state.current_question,
                st.session_state.current_answer,
                combined_feedback,
                correct_answer,
                is_correct,
                st.session_state.response_time
            )
            
            # チャット履歴に追加
            now = datetime.datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append({
                "type": "user",
                "content": st.session_state.current_question,
                "timestamp": now
            })
            st.session_state.chat_history.append({
                "type": "bot",
                "content": st.session_state.current_answer,
                "timestamp": now
            })
            
            # フォーム状態をリセット
            st.session_state.current_question = ""
            st.session_state.current_answer = ""
            st.session_state.feedback_given = True

            # 質問入力欄をリセット
            if "question_input" in st.session_state:
                st.session_state.question_input = ""
            
            st.success("フィードバックをいただきありがとうございます！")
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- 履歴閲覧ページのUI ---
def display_history_page():
    """履歴閲覧ページのUIを表示する"""
    # LINE風スタイルを適用
    apply_line_style()
    
    st.subheader("💬 トーク履歴")
    history_df = get_chat_history()

    if history_df.empty:
        st.info("まだトーク履歴がありません。")
        return

    # タブでセクションを分ける
    tab1, tab2 = st.tabs(["トーク履歴", "分析レポート"])

    with tab1:
        display_history_list(history_df)

    with tab2:
        display_metrics_analysis(history_df)

def display_history_list(history_df):
    """履歴リストをLINE風トーク画面で表示"""
    st.markdown("#### トーク履歴")
    
    # フィルターオプション
    filter_options = {
        "すべて表示": None,
        "👍 正確!": 1.0,
        "🤔 まあまあ": 0.5,
        "👎 いまいち": 0.0
    }
    
    # HTMLで直接カスタムスタイルのラジオボタンを作成
    st.markdown("""
    <style>
    .custom-filter {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    .filter-btn {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        cursor: pointer;
        border: 1px solid #ddd;
        background-color: #f9f9f9;
        transition: all 0.3s;
    }
    .filter-btn:hover {
        background-color: #eee;
    }
    .filter-btn.active {
        background-color: #F5F5F5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .filter-all { color: #000000; }
    .filter-good { color: #06C755; }
    .filter-neutral { color: #FFA500; }
    .filter-bad { color: #FF4500; }
    </style>
    """, unsafe_allow_html=True)

    # セッション状態で選択を管理
    if "filter_option" not in st.session_state:
        st.session_state.filter_option = "すべて表示"
    
    # ラジオボタン代わりのクリックボタン
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("すべて表示", key="filter_all"):
            st.session_state.filter_option = "すべて表示"
            st.rerun()
    
    with col2:
        if st.button("👍 正確!", key="filter_good"):
            st.session_state.filter_option = "👍 正確!"
            st.rerun()
    
    with col3:
        if st.button("🤔 まあまあ", key="filter_neutral"):
            st.session_state.filter_option = "🤔 まあまあ"
            st.rerun()
    
    with col4:
        if st.button("👎 いまいち", key="filter_bad"):
            st.session_state.filter_option = "👎 いまいち"
            st.rerun()
    
    # 現在選択中のフィルターを表示
    st.caption(f"現在のフィルター: {st.session_state.filter_option}")
    
    # フィルター適用
    display_option = st.session_state.filter_option

    filter_value = filter_options[display_option]
    if filter_value is not None:
        filtered_df = history_df[history_df["is_correct"].notna() & (history_df["is_correct"] == filter_value)]
    else:
        filtered_df = history_df

    if filtered_df.empty:
        st.info("選択した条件に一致する履歴はありません。")
        return

    # グループ化して日付ごとに表示
    filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
    unique_dates = filtered_df['date'].unique()
    
    # ページネーション
    dates_per_page = 1  # 1日分ずつ表示
    total_pages = len(unique_dates)
    
    current_page = st.number_input('ページ', min_value=1, max_value=max(1, total_pages), value=1, step=1)
    start_idx = (current_page - 1) % total_pages
    
    # 選択された日付のデータを表示
    if start_idx < len(unique_dates):
        selected_date = unique_dates[start_idx]
        date_df = filtered_df[filtered_df['date'] == selected_date]
        
        # LINE風のチャット表示
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 日付の区切り線を表示
        display_date_divider(selected_date.strftime("%Y年%m月%d日"))
        
        # その日の会話を時系列で表示
        for _, row in date_df.iterrows():
            # 時刻だけ抽出
            time_str = pd.to_datetime(row['timestamp']).strftime("%H:%M")
            
            # ユーザーの質問
            display_user_message(row['question'], time_str)
            
            # AIの回答
            display_bot_message(row['answer'], time_str)
            
            # フィードバック情報がある場合
            if pd.notna(row['feedback']):
                with st.expander("📝 フィードバック詳細"):
                    st.markdown(f"**評価:** {row['feedback']}")
                    if pd.notna(row['correct_answer']) and row['correct_answer'] != "":
                        st.markdown(f"**提案された回答:** {row['correct_answer']}")
                    
                    # 評価指標の表示
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("正確性", f"{row['is_correct']:.1f}")
                    metrics_cols[1].metric("応答時間", f"{row['response_time']:.2f}秒")
                    metrics_cols[2].metric("単語数", f"{row['word_count']}")
                    metrics_cols[3].metric("BLEU", f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.caption(f"{total_pages}日分中 {start_idx+1}日目を表示")
    else:
        st.info("表示できるデータがありません。")

# 以下の関数は基本的な機能は変えずにスタイルだけLINE風に変更

def display_metrics_analysis(history_df):
    """評価指標の分析結果を表示する"""
    # LINE風スタイルを適用済み
    st.write("#### 会話の分析レポート")

    # is_correct が NaN のレコードを除外して分析
    analysis_df = history_df.dropna(subset=['is_correct'])
    if analysis_df.empty:
        st.warning("分析可能なデータがありません。")
        return

    # 評価ラベルをLINEらしく変更
    accuracy_labels = {1.0: '👍 正確!', 0.5: '🤔 まあまあ', 0.0: '👎 いまいち'}
    analysis_df['評価'] = analysis_df['is_correct'].map(accuracy_labels)

    # 正確性の分布
    st.write("##### フィードバック分布")
    accuracy_counts = analysis_df['評価'].value_counts()
    if not accuracy_counts.empty:
        st.bar_chart(accuracy_counts)
    else:
        st.info("評価データがありません。")

    # 以下の分析部分は基本的な機能を維持
    # 応答時間と他の指標の関係
    st.write("##### 応答時間とその他の指標の関係")
    metric_options = ["bleu_score", "similarity_score", "relevance_score", "word_count"]
    valid_metric_options = [m for m in metric_options if m in analysis_df.columns and analysis_df[m].notna().any()]

    if valid_metric_options:
        metric_option = st.selectbox(
            "比較する評価指標を選択",
            valid_metric_options,
            key="metric_select"
        )

        chart_data = analysis_df[['response_time', metric_option, '評価']].dropna()
        if not chart_data.empty:
             st.scatter_chart(
                chart_data,
                x='response_time',
                y=metric_option,
                color='評価',
            )
        else:
            st.info(f"選択された指標 ({metric_option}) と応答時間の有効なデータがありません。")

    else:
        st.info("応答時間と比較できる指標データがありません。")

    # 全体の評価指標の統計
    st.write("##### 評価指標の統計")
    stats_cols = ['response_time', 'bleu_score', 'similarity_score', 'word_count', 'relevance_score']
    valid_stats_cols = [c for c in stats_cols if c in analysis_df.columns and analysis_df[c].notna().any()]
    if valid_stats_cols:
        metrics_stats = analysis_df[valid_stats_cols].describe()
        st.dataframe(metrics_stats)
    else:
        st.info("統計情報を計算できる評価指標データがありません。")

    # 正確性レベル別の平均スコア
    st.write("##### 評価レベル別の平均スコア")
    if valid_stats_cols and '評価' in analysis_df.columns:
        try:
            accuracy_groups = analysis_df.groupby('評価')[valid_stats_cols].mean()
            st.dataframe(accuracy_groups)
        except Exception as e:
            st.warning(f"評価別スコアの集計中にエラーが発生しました: {e}")
    else:
         st.info("評価レベル別の平均スコアを計算できるデータがありません。")

    # カスタム評価指標：効率性スコア
    st.write("##### 効率性スコア (正確性 / (応答時間 + 0.1))")
    if 'response_time' in analysis_df.columns and analysis_df['response_time'].notna().any():
        analysis_df['efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0) + 0.1)
        if 'id' in analysis_df.columns:
            top_efficiency = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
            if not top_efficiency.empty:
                st.bar_chart(top_efficiency.set_index('id')['efficiency_score'])
            else:
                st.info("効率性スコアデータがありません。")
        else:
             st.bar_chart(analysis_df.sort_values('efficiency_score', ascending=False).head(10)['efficiency_score'])
    else:
        st.info("効率性スコアを計算するための応答時間データがありません。")

# --- サンプルデータ管理ページのUI ---
def display_data_page():
    """サンプルデータ管理ページのUIを表示する"""
    # LINE風スタイルを適用
    apply_line_style()
    
    st.subheader("🗂 データ管理")
    count = get_db_count()
    st.write(f"現在のデータベースには {count} 件のトーク履歴があります。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("サンプルデータを追加", key="create_samples"):
            create_sample_evaluation_data()
            st.rerun()

    with col2:
        if st.button("データベースをクリア", key="clear_db_button"):
            if clear_db():
                st.rerun()

    # 評価指標に関する解説
    st.subheader("評価指標の説明")
    metrics_info = get_metrics_descriptions()
    for metric, description in metrics_info.items():
        with st.expander(f"{metric}"):
            st.write(description)