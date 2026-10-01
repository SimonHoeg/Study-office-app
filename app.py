"""Week 6 at the study office: who should the advisers talk to?

The model sits next to the code, in model/: booster.json (XGBoost) + preprocess.json (scaling and one-hot as plain
numbers), config.json (features, metrics, costs) and README.md (model card). They are written by the notebook
"final assignment 4" (step 9). The students are read straight from GitHub, like the notebook does.
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

MODEL_DIR = Path(__file__).parent / "model"
URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"
LOCAL = Path(__file__).parent / "data"
BACKGROUND = ["programme", "gender", "age", "international", "married", "admission_grade", "first_gen",
              "moved_from_home", "evening_programme"]
LABEL = {"fees_owed": "owes fees", "submitted_share": "few assignments handed in",
         "missed_last3": "missed recent assignments", "logins_last3": "few logins lately", "logins_total": "few logins overall",
         "logins_trend": "logins falling", "quiz_mean": "low quiz scores", "weeks_since_login": "not logged in lately",
         "programme": "programme", "su_scholarship": "no SU", "age": "age", "gender": "gender",
         "admission_grade": "admission grade", "international": "international", "first_gen": "first in family",
         "married": "married", "moved_from_home": "moved from home", "evening_programme": "evening programme"}

st.set_page_config(page_title="Week 6 at the study office", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- model and students, loaded once per server
@st.cache_resource
def load():
    model = Model(MODEL_DIR)                         # no pickle: loads with any recent pandas/xgboost
    config = json.load(open(MODEL_DIR / "config.json"))
    card = open(MODEL_DIR / "README.md").read().split("---", 2)[-1].replace("\n# ", "\n#### ")
    base = str(LOCAL) + "/" if (LOCAL / "new_week6.csv").exists() else URL
    history = pd.read_csv(base + "history_week6.csv")
    new = pd.read_csv(base + "new_week6.csv")
    features = config["numeric_features"] + config["categorical_features"]

    past = history[history["cohort"] == 2025].copy()  # the year the model never trained on
    past["risk"] = model.predict_proba(past[features])

    new["risk"] = model.predict_proba(new[features])
    contrib = model.contributions(new[features])
    new["why"] = [" · ".join(LABEL.get(f, f) for f, v in row.sort_values(ascending=False).head(3).items() if v > 0)
                  for _, row in contrib.iterrows()]
    push = contrib.clip(lower=0)
    new["from_background"] = (push[BACKGROUND].sum(axis=1) / push.sum(axis=1)).fillna(0)
    new["offer"] = new.apply(offer, axis=1)
    new = new.sort_values("risk", ascending=False).reset_index(drop=True)
    new.insert(0, "rank", np.arange(1, len(new) + 1))
    return model, config, card, past, new


def offer(s):
    """A first suggestion for the kind of conversation (an adviser decides)."""
    if s["fees_owed"] == 1:
        return "💰 money: fees / SU guidance"
    if s["weeks_since_login"] >= 1 or s["logins_last3"] <= 5:
        return "📞 personal check-in: gone quiet"
    if s["submitted_share"] <= 0.5 or s["missed_last3"] >= 2:
        return "📚 study plan / study group"
    return "👋 general check-in"


def boxes(data, contacted):
    """The four boxes, precision and recall for a set of contacted students."""
    left = data["left"] == 1
    tp, fp = int((contacted & left).sum()), int((contacted & ~left).sum())
    fn, tn = int((~contacted & left).sum()), int((~contacted & ~left).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan")}


def contacted_by_rule(data, rule, value):
    if rule == "number of conversations":
        return data["risk"].rank(ascending=False, method="first") <= value
    return data["risk"] >= value


def net_value(b, talk, worry, leave, helps):
    return b["TP"] * helps * leave - (b["TP"] + b["FP"]) * talk - b["FP"] * worry


def pct(x):
    return "–" if pd.isna(x) else f"{x:.0%}"


model, config, card, past, new = load()

# ---------------------------------------------------------------- sidebar: the rule
with st.sidebar:
    st.header("🧭 The rule")
    rule = st.radio("Contact students by", ["number of conversations", "risk cut-off"],
                    help="With three advisers the office can hold about 40 conversations at week 6.")
    if rule == "number of conversations":
        value = st.slider("Conversations this week", 5, 200, int(config.get("capacity", 40)), 1)
        rule_text = f"the {value} highest risks"
    else:
        value = st.slider("Contact everyone with a risk of at least", 0.02, 0.90, 0.30, 0.01, format="%.2f")
        rule_text = f"everyone with a risk of at least {value:.0%}"
    st.caption(f"Model: XGBoost · validation AUC {config['metrics_validation']['auc']} · "
               f"trained on {config['periods']['train']}, checked on {config['periods']['validation']}")
    st.info("The list only suggests whom to **offer** a conversation. An adviser decides, and students can say no.")

st.title("🎓 Week 6 at the study office")
st.markdown(f"Rule in use: **contact {rule_text}**. Change it in the sidebar.")
tab_list, tab_mistakes, tab_groups, tab_costs, tab_hood = st.tabs(
    ["📋 This week's list", "⚖️ The mistakes of the rule", "🌍 Per group", "💶 Costs", "⚙️ About the model"])

# ---------------------------------------------------------------- 1 · this week's list
with tab_list:
    new["contact"] = contacted_by_rule(new, rule, value)
    n = int(new["contact"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Students this week", len(new))
    c2.metric("On the list", n)
    c3.metric("Expected to leave among them", f"≈ {new.loc[new['contact'], 'risk'].sum():.0f}",
              help="The sum of their risks. The model over-predicted slightly in 2025, so read it as an upper estimate.")

    only = st.toggle("Show only the students on the list", value=True)
    show = new[new["contact"]] if only else new
    st.dataframe(
        show[["rank", "contact", "student_id", "risk", "programme", "international", "fees_owed", "submitted_share",
              "missed_last3", "logins_last3", "weeks_since_login", "why", "offer", "from_background"]],
        column_config={
            "rank": "#", "contact": st.column_config.CheckboxColumn("on list"), "student_id": "student",
            "risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="percent"),
            "international": st.column_config.CheckboxColumn("intl."), "fees_owed": st.column_config.CheckboxColumn("owes fees"),
            "submitted_share": st.column_config.NumberColumn("handed in", format="percent"),
            "missed_last3": "missed (3 wks)", "logins_last3": "logins (3 wks)", "weeks_since_login": "wks since login",
            "why": "main reasons (model)", "offer": "suggested conversation",
            "from_background": st.column_config.ProgressColumn(
                "push from background", min_value=0, max_value=1, format="percent",
                help="How much of the push towards a high risk comes from who the student is (programme, age, "
                     "gender…) rather than what they do. Above 50 %: an adviser should look twice before contacting.")},
        hide_index=True, width="stretch", height=520)
    mostly_bg = int((new["contact"] & (new["from_background"] > 0.5)).sum())
    if mostly_bg:
        st.warning(f"⚠️ {mostly_bg} of the {n} students are on the list mostly because of their background, "
                   "not their behaviour. Look at them before contacting.")
    st.download_button("⬇️ Download the list (CSV)", new[new["contact"]].drop(columns=["contact"]).to_csv(index=False),
                       "week6_list.csv", "text/csv")

# ---------------------------------------------------------------- 2 · the mistakes of the rule, on 2025
with tab_mistakes:
    past["contact"] = contacted_by_rule(past, rule, value)
    b = boxes(past, past["contact"])
    st.markdown(f"If the office had used this rule on the **2025 students** ({len(past)} students, "
                f"{int(past['left'].sum())} of whom left later), this is what would have happened:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 reached in time", b["TP"], help="contacted, and would have left (true positive)")
    c2.metric("📞 worried for nothing", b["FP"], help="contacted, but would have stayed anyway (false positive)")
    c3.metric("🚪 missed", b["FN"], help="not contacted, and left (false negative)")
    c4.metric("✅ left alone, stayed", b["TN"], help="not contacted, and stayed (true negative)")
    st.markdown(f"> **{b['TP']} students reached in time, {b['FP']} worried for nothing, {b['FN']} missed.**")
    c1, c2 = st.columns(2)
    c1.metric("Precision", pct(b["precision"]),
              help="Of the students we contacted, the share who were really at risk.")
    c1.caption(f"Of the {b['TP'] + b['FP']} students contacted, {b['TP']} were really at risk.")
    c2.metric("Recall", pct(b["recall"]), help="Of the students who left, the share we reached.")
    c2.caption(f"Of the {b['TP'] + b['FN']} students who left, we reached {b['TP']}.")

    st.markdown("#### How the mistakes move with the number of conversations")
    ks = np.arange(5, 301, 5)
    curve = pd.DataFrame([boxes(past, past["risk"].rank(ascending=False, method="first") <= k) for k in ks], index=ks)
    curve = curve.rename(columns={"TP": "reached in time", "FP": "worried for nothing", "FN": "missed"})
    st.line_chart(curve[["reached in time", "worried for nothing", "missed"]], x_label="conversations",
                  y_label="students")
    st.caption("More conversations: fewer missed students, but many more worried for nothing. No rule removes both.")

# ---------------------------------------------------------------- 3 · per group
with tab_groups:
    st.markdown("The same rule, on the 2025 students, split into **domestic** and **international** students.")
    rows = []
    for g, name in [(0, "domestic"), (1, "international")]:
        part = past[past["international"] == g]
        bb = boxes(part, part["contact"])
        rows.append({"group": name, "students": len(part), "share who left": part["left"].mean(),
                     "mean predicted risk": part["risk"].mean(), "contacted": bb["TP"] + bb["FP"],
                     "reached in time": bb["TP"], "worried for nothing": bb["FP"], "missed": bb["FN"],
                     "precision": bb["precision"], "recall": bb["recall"]})
    groups = pd.DataFrame(rows).set_index("group")
    st.dataframe(groups.style.format({"share who left": "{:.1%}", "mean predicted risk": "{:.1%}",
                                      "precision": "{:.0%}", "recall": "{:.0%}"}, na_rep="–"), width="stretch")
    d, i = groups.loc["domestic"], groups.loc["international"]
    st.markdown(f"""
- A domestic student who leaves is reached **{pct(d['recall'])}** of the time, an international student **{pct(i['recall'])}**.
- Both groups leave about equally often, but **international students log in about a third less**
  ({past.loc[past['international'] == 1, 'logins_total'].mean():.0f} vs
  {past.loc[past['international'] == 0, 'logins_total'].mean():.0f} logins in weeks 1–6). Few logins mean less for
  them, so a rule that leans on logins treats the groups differently.
- Only {int(past.loc[past['international'] == 1, 'left'].sum())} international students left in 2025: these numbers are
  uncertain. Check them every year before using the rule.
""")

# ---------------------------------------------------------------- 4 · our own addition: costs
with tab_costs:
    st.markdown("What is the rule worth? The numbers below are **assumptions**, and choosing them is a policy decision "
                "for the university, not for the model. Change them and see how the best rule moves.")
    costs = config.get("costs_dkk", {})
    c1, c2, c3, c4 = st.columns(4)
    talk = c1.number_input("A conversation (DKK)", 0, 10_000, int(costs.get("conversation", 500)), 100)
    worry = c2.number_input("A worried student (DKK)", 0, 50_000, int(costs.get("false_alarm", 2_000)), 500)
    leave = c3.number_input("A student who leaves (DKK)", 0, 300_000, int(costs.get("student_leaves", 60_000)), 5_000)
    helps = c4.slider("Share a conversation keeps", 0.0, 1.0, float(costs.get("share_helped", 0.30)), 0.05)

    breakeven = (talk + worry) / (helps * leave + worry) if helps * leave + worry else float("nan")
    cuts = np.round(np.arange(0.02, 0.905, 0.01), 2)
    values = pd.Series([net_value(boxes(past, past["risk"] >= c), talk, worry, leave, helps) for c in cuts], index=cuts)
    best_cut = float(values.idxmax())
    best_n = int((past["risk"] >= best_cut).sum())
    now = net_value(boxes(past, past["contact"]), talk, worry, leave, helps)

    c1, c2, c3 = st.columns(3)
    c1.metric("Break-even risk", "–" if pd.isna(breakeven) else f"{breakeven:.1%}", help="Contact a student when the risk is above this.")
    c2.metric("Best cut-off on 2025", f"{best_cut:.0%}", f"{best_n} students", delta_color="off")
    c3.metric("Your current rule is worth", f"{now:,.0f} DKK")
    st.line_chart(pd.DataFrame({"net value (DKK)": values}), x_label="cut-off", y_label="DKK on 2025")

    in40 = boxes(past, past["risk"].rank(ascending=False, method="first") <= 40)
    in53 = boxes(past, past["risk"].rank(ascending=False, method="first") <= 53)
    extra = net_value(in53, talk, worry, leave, helps) - net_value(in40, talk, worry, leave, helps)
    st.markdown(f"""
**What would a fourth adviser buy?** About 13 more conversations (40 → 53) would have reached
**{in53['TP'] - in40['TP']} more students** who left in 2025, worth **{extra:,.0f} DKK** under these assumptions.
{"The best rule contacts more students than three advisers can handle: more capacity pays off." if best_n > 40 else
 "The best rule contacts fewer than 40 students: under these assumptions, capacity is not the bottleneck."}
""")

# ---------------------------------------------------------------- 5 · about the model
with tab_hood:
    left, right = st.columns(2)
    with left:
        st.markdown("### Model card")
        st.markdown(card)
    with right:
        st.markdown("### config.json")
        st.json({k: config[k] for k in ["version", "periods", "metrics_validation", "capacity", "costs_dkk",
                                        "dropped_as_leakage", "versions"] if k in config})
        st.markdown("""
### Use it responsibly
- The list is a **suggestion**: an adviser decides whom to contact, and every student can say no.
- Tell students that the office uses data to offer help, and let them contest it (GDPR).
- Education is a **high-risk** area under the EU AI Act: keep a human in the loop, document the model, and check
  its mistakes per group every year.
""")
