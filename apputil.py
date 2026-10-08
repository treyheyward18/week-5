import plotly.express as px
import pandas as pd

# update/add code below ...

DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/"
    "datasets/main/data/titanic.csv"
)

AGE_GROUPS = ["Child", "Teen", "Adult", "Senior"]
CLASS_LABELS = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}


def to_snake_case(text):
    """Convert a column name like 'Age Group' into 'age_group'."""
    return "_".join(str(text).strip().lower().split())


def load_titanic(url=DATA_URL):
    """Load the Titanic dataset with lowercase-underscore column names."""
    data = pd.read_csv(url)
    data.columns = [to_snake_case(col) for col in data.columns]
    return data


df = load_titanic()


# ---------------------------------------------------------------------------
# Exercise 1: Survival Patterns
# ---------------------------------------------------------------------------

def add_age_group(data):
    """Return a copy of `data` with a categorical `age_group` column.

    Child: up to 12, Teen: 13-19, Adult: 20-59, Senior: 60+.
    Passengers with a missing age get a missing age group.
    """
    data = data.copy()
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 12, 19, 59, float("inf")],
        labels=AGE_GROUPS,
        right=True,
        include_lowest=True,
    )
    return data


def survival_demographics():
    """Summarize survival by passenger class, sex, and age group.

    Returns a DataFrame with one row for every combination of class,
    sex, and age group (including combinations with no passengers) and
    the columns `n_passengers`, `n_survivors`, and `survival_rate`.
    """
    data = add_age_group(df)
    data["pclass"] = pd.Categorical(data["pclass"], categories=[1, 2, 3])
    data["sex"] = pd.Categorical(
        data["sex"], categories=["female", "male"]
    )

    summary = (
        data.groupby(["pclass", "sex", "age_group"], observed=False)
        .agg(
            n_passengers=("survived", "size"),
            n_survivors=("survived", "sum"),
        )
        .reset_index()
    )
    summary["pclass"] = summary["pclass"].astype(int)
    summary["sex"] = summary["sex"].astype(str)
    summary["n_survivors"] = summary["n_survivors"].astype(int)
    summary["survival_rate"] = (
        summary["n_survivors"] / summary["n_passengers"]
    ).fillna(0.0)

    return summary.sort_values(["pclass", "sex", "age_group"]).reset_index(
        drop=True
    )


def visualize_demographic():
    """Did women and children survive more often than men in every class?

    Grouped bar chart: survival rate by age group, split by sex, with one
    panel per passenger class.
    """
    summary = survival_demographics()
    summary = summary[summary["n_passengers"] > 0].copy()
    summary["class"] = summary["pclass"].map(CLASS_LABELS)

    fig = px.bar(
        summary,
        x="age_group",
        y="survival_rate",
        color="sex",
        barmode="group",
        facet_col="class",
        category_orders={
            "age_group": AGE_GROUPS,
            "class": list(CLASS_LABELS.values()),
            "sex": ["female", "male"],
        },
        hover_data=["n_passengers", "n_survivors"],
        labels={
            "age_group": "Age group",
            "survival_rate": "Survival rate",
            "sex": "Sex",
            "class": "Class",
        },
        title="Survival rate by age group and sex, within each class",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


# ---------------------------------------------------------------------------
# Exercise 2: Family Size and Wealth
# ---------------------------------------------------------------------------

def add_family_size(data):
    """Return a copy of `data` with a `family_size` column."""
    data = data.copy()
    data["family_size"] = data["sibsp"] + data["parch"] + 1
    return data


def family_groups():
    """Summarize ticket fares by family size and passenger class.

    Returns a DataFrame with `n_passengers`, `avg_fare`, `min_fare`, and
    `max_fare` for each family size / class group, sorted by class and
    then family size.
    """
    data = add_family_size(df)
    summary = (
        data.groupby(["pclass", "family_size"])
        .agg(
            n_passengers=("fare", "size"),
            avg_fare=("fare", "mean"),
            min_fare=("fare", "min"),
            max_fare=("fare", "max"),
        )
        .reset_index()
    )
    return summary.sort_values(["pclass", "family_size"]).reset_index(
        drop=True
    )


def last_names():
    """Return the count of passengers for each last name as a Series."""
    names = df["name"].str.split(",").str[0].str.strip()
    return names.value_counts()


def visualize_families():
    """Do larger families pay more for their tickets, and does that
    depend on passenger class?

    Line chart of average fare by family size, one line per class.
    """
    summary = family_groups()
    summary["class"] = summary["pclass"].map(CLASS_LABELS)

    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="class",
        markers=True,
        category_orders={"class": list(CLASS_LABELS.values())},
        hover_data=["n_passengers", "min_fare", "max_fare"],
        labels={
            "family_size": "Family size (incl. passenger)",
            "avg_fare": "Average fare",
            "class": "Class",
        },
        title="Average ticket fare by family size and class",
    )
    fig.update_yaxes(tickprefix="$")
    fig.update_xaxes(dtick=1)
    return fig


# ---------------------------------------------------------------------------
# Bonus
# ---------------------------------------------------------------------------

def determine_age_division():
    """Add `older_passenger`: True if the passenger's age is above the
    median age of their passenger class, False otherwise.
    """
    data = df.copy()
    class_median = data.groupby("pclass")["age"].transform("median")
    data["older_passenger"] = data["age"] > class_median
    return data


def visualize_age_division():
    """Within each class, did older passengers survive less often?"""
    data = determine_age_division()
    data = data[data["age"].notna()]
    summary = (
        data.groupby(["pclass", "older_passenger"])["survived"]
        .mean()
        .reset_index(name="survival_rate")
    )
    summary["class"] = summary["pclass"].map(CLASS_LABELS)
    summary["age_division"] = summary["older_passenger"].map(
        {True: "Above class median age", False: "At or below median"}
    )

    fig = px.bar(
        summary,
        x="class",
        y="survival_rate",
        color="age_division",
        barmode="group",
        category_orders={"class": list(CLASS_LABELS.values())},
        labels={
            "class": "Class",
            "survival_rate": "Survival rate",
            "age_division": "Age within class",
        },
        title="Survival rate for older vs. younger passengers in each class",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def visualize_family_size():
    """Bonus chart used by app.py (same as the age-division chart)."""
    return visualize_age_division()
