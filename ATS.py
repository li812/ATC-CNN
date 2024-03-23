import os
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader

def ug_pg_miner(qualification):
    df = pd.read_csv('dataset/ug_pg.csv')
    #If qualification is UG
    if qualification in df['UG'].values:
        print("The qualification is an Undergraduate course.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        from joblib import dump, load
        model = load('KNN_Model/model.joblib')
        vectorizer = load('KNN_Model/vectorizer.joblib')
        query = vectorizer.transform([qualification])
        distances, indices = model.kneighbors(query)
        index = indices[0][0]
        # Append all PG corresponding to the UG in the dataset to a list Qualified_PG
        Qualified_PG = df[df['UG'] == df.loc[index, 'UG']]['PG'].tolist()
        pred_keywords = []
        directory = 'cv_base'
        pred_keywords.append(qualification)
        pred_keywords.extend(Qualified_PG)
        def search_cv(pred_keywords,directory):
            selected_resumes = []
            for filename in os.listdir(directory):
                if filename.endswith('.pdf'):
                    with open(os.path.join(directory, filename), 'rb') as file:
                        reader = PdfReader(file)

                        for page in reader.pages:
                            page_text = page.extract_text()

                            if any(keyword.lower() in page_text.lower() for keyword in pred_keywords):
                                selected_resumes.append(filename)
                                break
            return selected_resumes
        selected_CV = search_cv(pred_keywords,directory)
        return selected_CV


    #If qualification is PG
    elif qualification in df['PG'].values:
        print("The qualification is a Postgraduate course.")

        pred_keywords = []
        directory = 'cv_base'

        pred_keywords.append(qualification)

        def search_cv(pred_keywords,directory):
            selected_resumes = []
            for filename in os.listdir(directory):
                if filename.endswith('.pdf'):
                    with open(os.path.join(directory, filename), 'rb') as file:
                        reader = PdfReader(file)

                        for page in reader.pages:
                            page_text = page.extract_text()

                            if any(keyword.lower() in page_text.lower() for keyword in pred_keywords):
                                selected_resumes.append(filename)
                                break
            return selected_resumes
        selected_CV = search_cv(pred_keywords,directory)
        return selected_CV

    else:
        return False





import os
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader

# List of keywords to search for
def keyword_miner(pred_keywords):
    directory = 'cv_base'
    selected_CV = []
    
    def search_cv(pred_keywords, directory):
        selected_resumes = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                with open(os.path.join(directory, filename), 'rb') as file:
                    reader = PdfReader(file)
                    resume_contains_keyword = False  # Flag to check if the resume contains any keyword
                    for keyword in pred_keywords:
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if keyword.lower() in page_text.lower():
                                resume_contains_keyword = True
                                break  # No need to continue searching pages if keyword found
                        if resume_contains_keyword:
                            selected_resumes.append(filename)
                            break  # No need to check other keywords if one is found
        return selected_resumes

    selected_CV = search_cv(pred_keywords, directory)
    return selected_CV


def update_base():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from joblib import dump, load

    df = pd.read_csv('dataset/ug_pg.csv')

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['UG'])

    # Train a k-NN model
    model = NearestNeighbors(n_neighbors=1, metric='cosine')
    model.fit(X)

    # Save the model and vectorizer
    dump(model, 'KNN_Model/model.joblib')
    dump(vectorizer, 'KNN_Model/vectorizer.joblib')