import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier

movies = pd.read_csv('movies.csv', encoding = "ISO-8859-1")
grosses = movies["gross"]

# ------------------------ADJUSTING GROSS FOR INFLATION------------------------
def adjustForInflation(movies):
    years = movies["year"]
    grosses = movies["gross"]
    minYear = min(years)
    newGrosses = []
    for i in movies.index:
        yearsPast = years[i] - minYear
        newGrosses.append(grosses[i] * 1.03**(yearsPast))

    df = pd.DataFrame(newGrosses, columns = ['gross'])
    movies["gross"] = df["gross"]

# ------------------------FEATURIZING DIRECTORS------------------------
def featurizeDirectors(movies):
    directors = movies["director"]
    grosses = movies["gross"]
    # grabbing all directors
    directorDict = dict()
    for i in directors.index:
        if (directors[i] not in directorDict):
            directorDict[directors[i]] = [i]
        else:
            directorDict[directors[i]].append(i)

    # creating dictionary that has director name as key and their accumulated gross
    # as the value
    directorGross = dict()
    for i in movies["gross"].index:
        if (directors[i] not in directorGross):
            directorGross[directors[i]] = grosses[i]
        else:
            directorGross[directors[i]] += grosses[i]

    # replacing values in director column with gross
    for d in directorGross:
        movies['director'].replace(d, directorGross[d],inplace=True)

# ------------------------FEATURIZING ACTORS------------------------
def featurizeActors(movies):
    actors = movies["star"]
    grosses = movies["gross"]
    # grabbing all actors
    actorDict = dict()
    for i in actors.index:
        if (actors[i] not in actorDict):
            actorDict[actors[i]] = [i]
        else:
            actorDict[actors[i]].append(i)

    # creating dictionary that has actor name as key and their accumulated gross
    # as the value
    actorGross = dict()
    for i in movies["gross"].index:
        if (actors[i] not in actorGross):
            actorGross[actors[i]] = grosses[i]
        else:
            actorGross[actors[i]] += grosses[i]

    # replacing values in star column with gross
    for a in actorGross:
        movies['star'].replace(a, actorGross[a],inplace=True)

# ------------------------FEATURIZING COMPANIES------------------------
def featurizeCompanies(movies):
    companies = movies["company"]
    grosses = movies["gross"]
    # grabbing all companies
    companyDict = dict()
    for i in companies.index:
        if (companies[i] not in companyDict):
            companyDict[companies[i]] = [i]
        else:
            companyDict[companies[i]].append(i)

    # creating dictionary that has company name as key and their accumulated gross
    # as the value
    companyGross = dict()
    for i in movies["gross"].index:
        if (companies[i] not in companyGross):
            companyGross[companies[i]] = grosses[i]
        else:
            companyGross[companies[i]] += grosses[i]

    # replacing values in company column with gross
    for c in companyGross:
        movies['company'].replace(c, companyGross[c],inplace=True)

# ------------------------FEATURIZING WRITERS------------------------
def featurizeWriters(movies):
    writers = movies["writer"]
    # grabbing all writers
    writerDict = dict()
    for i in writers.index:
        if (writers[i] not in writerDict):
            writerDict[writers[i]] = [i]
        else:
            writerDict[writers[i]].append(i)

    # creating dictionary that has writer name as key and their accumulated gross
    # as the value
    writerGross = dict()
    for i in movies["gross"].index:
        if (writers[i] not in writerGross):
            writerGross[writers[i]] = grosses[i]
        else:
            writerGross[writers[i]] += grosses[i]

    # replacing values in writer column with gross
    for w in writerGross:
        movies['writer'].replace(w, writerGross[w],inplace=True)

# ------------------------FEATURIZING GENRES------------------------
def featurizeGenres(movies):
    genres = set(movies["genre"]) # grabbing unique set of genres
    allMovieGenres = movies["genre"] # grabbing each genre for each movie

    # add columns with different genres to movie data and, for each movie,
    # write down either 0 or 1 for its value
    # (ex. movie Texas Chainsaw Massacre will have isComedy = 0, isHorror = 1)
    for g in genres:
        isGenre = []
        for i in movies.index:
            if (allMovieGenres[i] == g):
                isGenre.append(1)
            else:
                isGenre.append(0)
        columnName = "is" + g
        df = pd.DataFrame(isGenre, columns = [columnName])
        movies[columnName] = df
    movies = movies.drop('genre', axis=1, inplace=True)

# ------------------------FEATURIZING RATING------------------------
def featurizeRating(movies):
    ratings = set(movies["rating"]) # grabbing unique set of ratings
    allMovieRatings = movies["rating"] # grabbing each rating for each movie

    # add columns with different ratings to movie data and, for each movie,
    # write down either 0 or 1 for its value
    # (ex. movie Lady Jane will have isPG-13 = 1, isG = 0)
    for r in ratings:
        isRating = []
        for i in movies.index:
            if (allMovieRatings[i] == r):
                isRating.append(1)
            else:
                isRating.append(0)
        columnName = "is" + r
        df = pd.DataFrame(isRating, columns = [columnName])
        movies[columnName] = df
    movies = movies.drop('rating', axis=1, inplace=True)


# ------------------------FEATURIZING COUNTRY------------------------
def featurizeCountry(movies):
    countries = set(movies["country"]) # grabbing unique set of ratings
    allMovieCountries = movies["country"] # grabbing each rating for each movie

    for c in countries:
        isCountry = []
        for i in movies.index:
            if (allMovieCountries[i] == c):
                isCountry.append(1)
            else:
                isCountry.append(0)
        columnName = "is" + c
        df = pd.DataFrame(isCountry, columns = [columnName])
        movies[columnName] = df
    movies = movies.drop('country', axis=1, inplace=True)

# ------------------------FEATURIZING RELEASE DATE------------------------
def featurizeDates(movies):
    allMovieDates = movies["released"] # grabbing each date for each movie
    months = []
    days = []

    for i in movies.index:
        date = allMovieDates[i].split("/")
        if (len(date)) < 3:
            date = date[0].split("-")
            if (len(date) == 1):
                year, month, day = date[0], 0, 0
            else:
                year, month, day = date[0], date[1], 0
        else:
            month, day, year = date[0], date[1], date[2]
        months.append(int(month))
        days.append(int(day))

    df = pd.DataFrame(months, columns = ["month"])
    movies["month"] = df
    df = pd.DataFrame(days, columns = ["day"])
    movies["day"] = df

    movies = movies.drop('released', axis=1, inplace=True)

# ------------------------FEATURIZE DATA------------------------
def featurizeData(movies):
    adjustForInflation(movies)
    featurizeDirectors(movies)
    featurizeActors(movies)
    featurizeCompanies(movies)
    featurizeWriters(movies)
    featurizeGenres(movies)
    featurizeRating(movies)
    featurizeDates(movies)
    featurizeCountry(movies)
featurizeData(movies)

y = movies['gross'].to_numpy()
movies.drop(['gross', 'name'], axis = 1, inplace=True)
X = movies.to_numpy()

# ------------------------LINEAR REGRESSION------------------------
def printAccuracyAndPredictor(movies, y, X):
    movies = pd.read_csv('movies.csv', encoding = "ISO-8859-1")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    learner = LinearRegression()
    accuracy = 0
    for train_indices, val_i in kf.split(X_train):
        learner.fit(X_train[train_indices], y_train[train_indices])
        accuracy += learner.score(X_train[val_i], y_train[val_i])
    accuracy /= 5
    print("Cross Validation Accuracy: ", accuracy)

    index = np.argmax(np.absolute(learner.coef_))
    columns = movies.columns.values
    print("Best predictor of success", columns[index])
    return learner
print("ACCURACY AND BEST PREDICTOR FOR LINEAR REG")
learner = printAccuracyAndPredictor(movies, y, X)

# ------------------------LOGISTIC REGRESSION------------------------
# create thresholds
def createThresholdsLogistic(movies):
    successes = []
    for i in movies.index:
        if grosses[i] < 12135679:
            successes.append(0)
        else:
            successes.append(1)
    return successes

def logisticRegression(movies, y, X):
    X = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    learner = LogisticRegression()
    accuracy = 0
    for train_indices, val_i in kf.split(X_train):
        learner.fit(X_train[train_indices], y_train[train_indices])
        accuracy += learner.score(X_train[val_i], y_train[val_i])
    accuracy /= 5
    print("Cross Validation Accuracy: ", accuracy)

    index = np.argmax(np.absolute(learner.coef_))
    columns = movies.columns.values
    print("Best predictor of success", columns[index])
    return learner
print("\nACCURACY AND BEST PREDICTOR FOR LOGISTIC REG")
binaryY = np.array(createThresholdsLogistic(movies))
logLearner = logisticRegression(movies, binaryY, X)

testMovies = {'avengers': np.array([356000000, 1205000000,743813007.0, 181, 8.5, 2702980274,
    617592, 915869198, 2019, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]).reshape(1,-1),
    'parasite': np.array([12000000, 10000000, 47000000, 132, 8.6, 21800000, 95779,
    47000000, 2019, 0, 0, 0, 0, 0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,11,8,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)}

# ------------------------PREDICT REVENUE OF NEW MOVIES------------------------
def predictMovieRevenue(movies, learner):
    # actual grosses
    grosses={'avengers': 858373000, 'parasite':19433690}

    # predict the grosses using our learner
    for movie in movies:
        prediction = learner.predict(movies[movie])[0]
        actual = grosses[movie]
        print("Estimated gross for ", movie, " is ", prediction)
        print("Actual gross for ", movie, " is ", actual)

        average = (prediction + actual)/2
        percentDiff = (max(actual,prediction) - min(prediction, actual)) / average
        print("Our prediction was ", percentDiff, " off")
print("\nPREDICTING REVENUE FOR RECENT MOVIES")
predictMovieRevenue(testMovies, learner)

# ------------------------PREDICT SUCCESS OF NEW MOVIES------------------------
def predictMovieSuccess(movies, learner):
    # actual grosses
    grosses={'avengers': 1, 'parasite':1}

    # predict the grosses using our learner
    for movie in movies:
        prediction = learner.predict(movies[movie])[0]
        actual = grosses[movie]
        print("Is ", movie, " a success? ", prediction)
        print("Our prediction was ", prediction == actual)
print("\nPREDICTING MOVIE SUCCESS")
predictMovieSuccess(testMovies, logLearner)

# ------------------------USING NEURAL NETWORKS------------------------
# create thresholds
def createThresholdsNeural(movies):
    successes = []
    for i in movies.index:
        if grosses[i] < 3447022:
            successes.append(0)
        elif grosses[i] > 26690101:
            successes.append(2)
        else:
            successes.append(1)
    df = pd.DataFrame(successes, columns = ['success'])
    movies["success"] = df["success"]
    movies.assign(success=df['success'])

createThresholdsNeural(movies)
#split data into training, validation, and testing
y = movies['success'].as_matrix()
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# try a few different architectures for the neural network
# NOTE: the last three take FOREVER to run, but (6000,6000,) was the best accuracy
#   If you want to run the last three, be sure to have the bigger lists uncommented.
shapes = [(1500,), (1500,1500,), (3000,)]
#shapes = [(1500,), (1500,1500,), (3000,), (3000,3000,), (6000,), (6000,6000,)]
accuracies = [0,0,0]
#accuracies = [0,0,0,0,0,0]
NNs = [MLPClassifier(hidden_layer_sizes=shape) for shape in shapes]
for s in range(len(shapes)):
    shape = shapes[s]
    NN = NNs[s]
    NN.fit(X_train, y_train)
    score = NN.score(X_validation, y_validation)
    accuracies[s] = score
    print("Architecture:", shape)
    print("\tValidation score:", score)

#get test accuracy for best neural network
best_index = accuracies.index(max(accuracies))
bestNN = NNs[best_index]
print("The best neural network has shape", shapes[best_index], "and has a validation accuracy of", accuracies[best_index])
testScore = bestNN.score(X_test, y_test)
print("The best neural network has a test accuracy of", testScore)

# predict success for best neural network
real = {'avengers':2, 'parasite':1}
for m in testMovies:
    pred = bestNN.predict(testMovies[m])[0]
    actual = real[m]
    print(m, "should be classified as", actual, "by the neural network.")
    print(m, "was classified as", pred)
