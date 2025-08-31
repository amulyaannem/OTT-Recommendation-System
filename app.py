import streamlit as st
import pandas as pd
import joblib

# Load data and model
combined_data = pd.read_csv('merged_ott_data_final.csv')
unrated = combined_data[combined_data['rating'].isnull()].copy()

model = joblib.load('model.pkl')
features = joblib.load('features.pkl')


unrated['predicted_rating'] = model.predict(unrated[features])

# App title
st.markdown(
    "<h1 style='text-align: center; color: #cc0066;'>🎥 OTT Movie Recommendations </h1>",
    unsafe_allow_html=True
)

# Pink floral background CSS
st.markdown("""
    <style>
    body {
        background-color: #ffe6f0;
        background-image: url('https://i.ibb.co/yFBXR7J/floral-bg.png');
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filter Recommendations")

platform_list = ['All'] + combined_data['platform'].unique().tolist()
selected_platform = st.sidebar.selectbox("Choose Platform:", platform_list)

desired_rating = st.sidebar.number_input("Desired Rating (optional)", min_value=1.0, max_value=7.0, step=0.5)
tolerance = st.sidebar.number_input("Tolerance (default 0.5)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)

st.sidebar.header("➕ Add a New Personal Rating")

new_title = st.sidebar.text_input("Movie Title")
new_platform = st.sidebar.selectbox("Platform for Rating", combined_data['platform'].unique())
new_rating = st.sidebar.number_input("Your Rating (1-7)", min_value=1.0, max_value=7.0, step=0.5)

if st.sidebar.button("Add Rating"):
    new_entry = pd.DataFrame({
        'title': [new_title],
        'platform': [new_platform],
        'rating': [new_rating]
    })
    new_entry.to_csv('my_rating.csv', mode='a', header=False, index=False)
    st.sidebar.success("✅ Rating added! (Retrain your model to reflect this.)")

if st.sidebar.button("Get Recommendations"):

    filtered = unrated.copy()

    if selected_platform != 'All':
        filtered = filtered[filtered['platform'] == selected_platform]

    if desired_rating > 0:
        filtered = filtered[
            (filtered['predicted_rating'] >= desired_rating - tolerance) &
            (filtered['predicted_rating'] <= desired_rating + tolerance)
        ]

    filtered = filtered.sort_values(by='predicted_rating', ascending=False).head(10)

    if filtered.empty:
        st.warning("⚠️ No movies found for the selected criteria. Try adjusting your filters.")
    else:
        for _, row in filtered.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"**Platform:** {row['platform']}")
            st.write(f"**Predicted Rating:** {round(row['predicted_rating'], 2)}")
            st.write("---")


else:
    st.subheader("📋 Top 10 Recommended Movies")
    top10 = unrated.sort_values(by='predicted_rating', ascending=False).head(10)
    
    import altair as alt

    chart_data = top10[['title', 'predicted_rating']].copy()
    c = alt.Chart(chart_data).mark_bar().encode(
        x='predicted_rating',
        y=alt.Y('title', sort='-x')
    )
    st.altair_chart(c, use_container_width=True)

    for _, row in top10.iterrows():
        st.markdown(f"### 🎬 {row['title']}")
        st.write(f"**Platform:** {row['platform']}")
        st.write(f"**Predicted Rating:** {round(row['predicted_rating'], 2)}")
        st.write("---")