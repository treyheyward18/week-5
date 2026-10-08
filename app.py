import streamlit as st

from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write(
'''
# Titanic Visualization 1

'''
)
st.write(
    "Did women and children survive more often than men "
    "in every passenger class?"
)
st.dataframe(survival_demographics())
# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2
'''
)
st.dataframe(family_groups())
st.write("Most common last names:")
st.dataframe(last_names().head(10))
st.write(
    "The last-name counts mostly agree with the family table, but not "
    "perfectly. The 7 passengers named Sage match the single 3rd-class "
    "group with a family size of 11 (only 7 of the 11 are in this data). "
    "But Andersson appears 9 times while the largest Andersson family has "
    "7 members: two unrelated passengers share the surname. A last name "
    "alone can overcount families, and it can't see family members who "
    "aren't in the dataset."
)
st.write(
    "Do larger families pay more for their tickets, and does that "
    "depend on passenger class?"
)
# Generate and display the figure
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus
'''
)
st.write(
    "Within each class, did passengers older than their class's median "
    "age survive less often?"
)
# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)
