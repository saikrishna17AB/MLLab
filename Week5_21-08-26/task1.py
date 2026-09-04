import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


FILE_NAME = "udemy_courses_with_student_ids_and_ratings.csv"
K = 2
TOP_N = 3

df = pd.read_csv(FILE_NAME)

print("=" * 70)
print("SMARTRECSYS - PERSONALIZED COURSE RECOMMENDER")
print("=" * 70)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

required_columns = [
    "student_id",
    "course_id",
    "course_title",
    "course_rating"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(
    subset=[
        "student_id",
        "course_id",
        "course_rating"
    ]
).copy()

df["course_rating"] = pd.to_numeric(
    df["course_rating"],
    errors="coerce"
)

df = df.dropna(
    subset=["course_rating"]
)

print("\nNumber of valid interaction records:", len(df))

print("\n" + "=" * 70)
print("TASK 1 - INTERACTION MATRIX ASSEMBLY")
print("=" * 70)
R = df.pivot_table(
    index="student_id",
    columns="course_id",
    values="course_rating",
    aggfunc="mean"
)

print("\nOriginal Interaction Matrix:")
print(R)

user_means = R.mean(axis=1)

print("\nStudent Baseline Mean Ratings:")
print(user_means)

R_centered = R.sub(
    user_means,
    axis=0
)

print("\nMean-Centered Matrix:")
print(R_centered)

R_centered = R_centered.fillna(0)

print("\nFinal Centered Interaction Matrix:")
print(R_centered)


print("\n" + "=" * 70)
print("TASK 2 - LOW-RANK FACTORIZATION VIA SVD")
print("=" * 70)


if min(R_centered.shape) < K:
    raise ValueError(
        f"Cannot use k={K}. "
        f"Interaction matrix shape is {R_centered.shape}."
    )


svd = TruncatedSVD(
    n_components=K,
    random_state=42
)

user_topic_times_sigma = svd.fit_transform(
    R_centered.values
)

singular_values = svd.singular_values_

V_transpose = svd.components_


Sigma = np.diag(
    singular_values
)


U = (
    user_topic_times_sigma
    / singular_values
)


V = V_transpose.T


print("\nTarget Rank k:", K)

print("\nUser-Topic Embedding Matrix U:")
print(U)

print("\nSingular Values:")
print(singular_values)

print("\nSigma:")
print(Sigma)

print("\nCourse-Topic Embedding Matrix V:")
print(V)

centered_prediction = (
    U
    @ Sigma
    @ V_transpose
)


predicted_ratings = (
    centered_prediction
    + user_means.values.reshape(-1, 1)
)


R_hat = pd.DataFrame(
    predicted_ratings,
    index=R.index,
    columns=R.columns
)


print("\nDense Reconstructed Rating Prediction Matrix:")
print(R_hat.round(3))


print("\n" + "=" * 70)
print("TASK 3 - COURSE RECOMMENDATION")
print("=" * 70)


def recommend_top_courses(student_id, top_n=3):
    """
    Recommend the top N courses that a student has not
    interacted with according to the reconstructed matrix.
    """

    if student_id not in R_hat.index:
        raise ValueError(
            f"Student '{student_id}' not found."
        )


    predicted_scores = R_hat.loc[
        student_id
    ].copy()


    already_interacted = R.loc[
        student_id
    ].dropna().index


    unseen_courses = predicted_scores.drop(
        labels=already_interacted,
        errors="ignore"
    )


    top_courses = (
        unseen_courses
        .sort_values(
            ascending=False
        )
        .head(top_n)
    )

    course_titles = (
        df[
            ["course_id", "course_title"]
        ]
        .drop_duplicates(
            subset=["course_id"]
        )
        .set_index("course_id")["course_title"]
    )


    recommendations = pd.DataFrame({
        "course_id": top_courses.index,
        "course_title": [
            course_titles.get(
                course_id,
                course_id
            )
            for course_id in top_courses.index
        ],
        "predicted_score": top_courses.values
    })


    recommendations["predicted_score"] = (
        recommendations["predicted_score"]
        .round(3)
    )


    return recommendations


student_id = df["student_id"].iloc[0]


recommendations = recommend_top_courses(
    student_id,
    TOP_N
)


print(f"\nStudent ID: {student_id}")

print("\nTop 3 Recommended Unseen Courses:")

if recommendations.empty:
    print("No unseen courses available for recommendation.")

else:
    print(
        recommendations.to_string(
            index=False
        )
    )


R.to_csv(
    "original_interaction_matrix.csv"
)

R_centered.to_csv(
    "centered_interaction_matrix.csv"
)

R_hat.to_csv(
    "predicted_rating_matrix.csv"
)

recommendations.to_csv(
    "top_course_recommendations.csv",
    index=False
)

print("\n" + "=" * 70)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nInteraction matrix shape:", R.shape)

print("SVD rank:", K)

print("\nOutput files:")
print("1. original_interaction_matrix.csv")
print("2. centered_interaction_matrix.csv")
print("3. predicted_rating_matrix.csv")
print("4. top_course_recommendations.csv")