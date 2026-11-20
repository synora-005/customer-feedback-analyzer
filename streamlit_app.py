"""The app: `streamlit run streamlit_app.py`"""
import streamlit as st

from app.core import MODEL_DIR, classify_comments, load_classifier, model_available

UNSURE_BELOW = 0.6      # under this confidence the app says the model is unsure

st.set_page_config(page_title="Customer feedback classifier")
st.title("Customer feedback classifier")
st.write("Type customer comments, one per line. A fine-tuned GPT-2 sorts each one into one of five "
         "categories: product issue, staff praise, staff complaint, pricing or store experience.")

if not model_available(MODEL_DIR):
    st.error(f"No trained model found in {MODEL_DIR}. Run `python train_it.py` first.")
    st.stop()


@st.cache_resource          # load the model once, not on every click
def get_model():
    return load_classifier(MODEL_DIR)


tokenizer, model = get_model()

text = st.text_area("Comments (one per line)", height=150,
                    placeholder="The cashier was so friendly.\nThe zip broke after two days.")

if st.button("Classify"):
    results = classify_comments(text.splitlines(), model, tokenizer)
    if not results:
        st.warning("Type at least one comment first.")
    else:
        st.dataframe(
            [{"Comment": r["comment"],
              "Category": r["label"],
              "Confidence": f"{r['confidence']:.0%}",
              "Note": "unsure" if r["confidence"] < UNSURE_BELOW else ""} for r in results],
            use_container_width=True, hide_index=True,
        )
        if len(results) == 1:
            st.bar_chart(results[0]["scores"])      # how likely the model thinks each category is

st.caption("Trained on 140 starter comments, so it can be wrong, sometimes with high confidence. "
           "Check anything important by hand.")