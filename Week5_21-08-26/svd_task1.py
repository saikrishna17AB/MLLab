import numpy as np
import pandas as pd

df = pd.read_csv("udemy_courses_with_student_ids_and_ratings.csv")

R = df.pivot_table(index="student_id", columns="course_id", values="course_rating")
studentmean = R.mean(axis=1)
globalmean = R.stack().mean()

studentmean = studentmean.fillna(globalmean)

Rcentered = R.subtract(studentmean, axis=0)
Rcentered = Rcentered.fillna(0)

A = Rcentered.to_numpy(dtype=float)


def truncatedSVD(A, k=2):
    m, n = A.shape
    if k < 1 or k > min(m, n):
        return None
    ATA = A.T @ A

    eigenvalues, eigenvectors = np.linalg.eigh(ATA)

    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    singularvalues = np.sqrt(np.maximum(eigenvalues, 0))

    singularvalues = singularvalues[:k]
    V = eigenvectors[:, :k]

    U = np.zeros((m, k))

    for i in range(k):
        if singularvalues[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / singularvalues[i]

    sigma = np.diag(singularvalues)
    Vt = V.T

    return U, sigma, Vt


# Perform trungated SVD
k = 2
U, sigma, Vt = truncatedSVD(A, k)

print(U)
print(sigma)
print(Vt)

Areconstructed = U @ sigma @ Vt
predictedarray = Areconstructed + studentmean.to_numpy().reshape(-1, 1)

predictedratings = pd.DataFrame(predictedarray, index=R.index, columns=R.columns)
print(predictedratings)

coursetitles = df.drop_duplicates("course_id").set_index("course_id")["course_title"]


def recommendedcourses(studentid, topn=3):
    if studentid not in predictedratings.index:
        return None
    studentprediction = predictedratings.loc[studentid]

    alreadyrated = set(df.loc[df["student_id"] == studentid, "course_id"])

    unseenpredictions = studentprediction.drop(
        labels=list(alreadyrated), errors="ignore"
    )

    unseenpredictions = unseenpredictions.sort_values(ascending=False)
    topcourses = unseenpredictions.head(topn)

    recommendations = pd.DataFrame(
        {
            "course_id": topcourses.index,
            "course_title": topcourses.index.map(coursetitles),
            "predicted_rating": topcourses.values,
        }
    )
    return recommendations


studentid = "STUDC8AMV4R"
recommedations = recommendedcourses(studentid, 4)
print(recommedations)
