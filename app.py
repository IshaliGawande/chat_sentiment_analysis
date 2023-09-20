import streamlit as st
#using streamlit you can deploy any machine learning model and any python project with ease and without worrying about the frontend.
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
#Seaborn is an amazing visualization library for statistical graphics plotting in Python.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
#Sentiment Analysis is the process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral.lexicon and simple rule-based model for sentiment analysis.

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
print(type(uploaded_file))
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    #UTF8 Decoder is a variable-length character decoding that can make any Unicode character readable. 
    df = preprocessor.preprocess(data)
    print(list(df))

    ##################
    df_sent = pd.DataFrame(df, columns=['date', 'user', 'message', 'only_date', 'year', 'month_num', 'month', 'day', 'day_name', 'hour', 'minute', 'period'])
    # #df['Date'] = pd.to_datetime(df['Date'])

    #df_sent = df.dropna()
    sentiments = SentimentIntensityAnalyzer()
    df_sent["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in df_sent["message"]]
    df_sent["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in df_sent["message"]]
    df_sent["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df_sent["message"]]
    print(df_sent.head())
    x = sum(df_sent["Positive"])
    y = sum(df_sent["Negative"])
    z = sum(df_sent["Neutral"])

    def Sentiment_score(a, b, c):
        if (a>b) and (a>c):
            return ("Positive 😊 ")
        elif (b>a) and (b>c):
            return ("Negative 😠 ")
        else:
            return ("Neutral 🙂 ")
    sentiment_score=Sentiment_score(x, y, z)
    ##################

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4  = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)
       

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)


        #Sentiment Analysis
        st.title("Sentiment Analysis")

        col1,col2 = st.columns(2)

        with col1:
             st.header("Sentiment Score")
             st.title(sentiment_score)
            
        with col2:
             st.header("")
             
            
       











