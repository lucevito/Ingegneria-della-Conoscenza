import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity   
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#Function that print the Top 10 most similar anime using Collaborative Filtering and Content based filtering 
def anime_similarity(anime_pivot,anime_data,genres_list,anime_index, title):

    time_sec =time.time()
    anime_matrix = csr_matrix(anime_pivot.values)
    # using the k-nearest neighbors algorithm
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(anime_matrix)
    distances, indices = model_knn.kneighbors(anime_pivot.loc
                        [anime_pivot.index == title].values.reshape(1, -1), n_neighbors = 11)

    # print the top 10 most similar anime
    print('\nUsing Collaborative Filtering, the Top 10 most similar anime are :')
    print('Warning : to reduce the use of ram,')
    print('the size of the database will be reduced, anime with at least 500 votes will be considered\n')
    for i in range(0, len(distances.flatten())):
        print('{0}: {1}, with distance of {2}:'
            .format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    print('Time to calculate it : {:0.2f} seconds\n'.format(time.time()-time_sec))

    print('Warning : to reduce the use of ram, the database size can be reduced')
    print('based on the type of anime such as tv movies etc ..')
    while True:
        print('\nEnter 0 for seach with all type')
        print('Enter 1 for seach with type Movie')
        print('Enter 2 for seach with type Music')
        print('Enter 3 for seach with type ONA')
        print('Enter 4 for seach with type OVA')
        print('Enter 5 for seach with type Special')
        print('Enter 6 for seach with type TV')
        type_select = input('Enter your choice : ')
        if type_select=='0'or type_select=='1'or type_select=='2'or type_select=='3'or type_select=='4'or type_select=='5'or type_select=='6':
            break
    time_sec =time.time()  
    # using the term frequency–inverse document frequency
    tf = TfidfVectorizer(analyzer='word')
    tf_matrix = tf.fit_transform(genres_list[int(type_select)])
    cosine_sim = cosine_similarity(tf_matrix, tf_matrix)

    idx = anime_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # scores of the 10 most similar anime
    sim_scores = sim_scores[1:11]
    # anime indices
    anime_indices = [i[0] for i in sim_scores]
    # print the top 10 most similar anime
    print('\nUsing Content based filtering, the Top 10 most similar anime are :\n')
    print(pd.DataFrame({'Anime name': anime_data['name'].iloc[anime_indices].values,
                                    'Rating': anime_data['rating'].iloc[anime_indices].values}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')


# Function that print the most similar anime using
# Random Forest Classifier ,  Support Vector Classifier , 
# Bagging classifier using k-nearest neighbors vote and
# Bagging classifier using Decision Tree Classifier
def anime_genre_like(anime_select,test):
    #column of features
    features = anime_select.columns[6:]
    #column of anime id
    y = anime_select['anime_id']
    print('Processing data...')
    time_sec =time.time()
    clf = RandomForestClassifier(n_jobs=10, random_state=42, max_depth=10)
    clf.fit(anime_select[features], y)
    print('Random Forest Classifier recommendation: ')
    print('Warning : the maximum depth of the tree is limited to 10 in this particular test\n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == clf.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del clf

    time_sec =time.time()
    svc = SVC()
    print('Processing data...')
    svc.fit(anime_select[features], y)
    print('Support Vector Classifier recommendation: \n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == svc.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del svc

    time_sec =time.time()  
    model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
    print('Processing data...')
    model.fit(anime_select[features],y)
    print('Bagging classifier using k-nearest neighbors vote recommendation: \n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == model.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del model

    time_sec =time.time()
    model=BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),random_state=0,n_estimators=100)
    print('Processing data...')
    model.fit(anime_select[features],y)
    print('Bagging classifier using Decision Tree Classifier recommendation: ')
    print('Warning : the maximum depth of the tree is limited to 10 in this particular test\n')
    print(pd.DataFrame({'Anime name' : anime_select['name'].loc[anime_select['anime_id'] == model.predict(test[features])[0]]}))
    print('Time to calculate it : {:0.2f} seconds'.format(time.time()-time_sec))
    print('\n')
    del model

    del anime_select

