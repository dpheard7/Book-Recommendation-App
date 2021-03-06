import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template, request, jsonify, render_template_string
from scipy import stats
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# TODO Remove commented out plt.show() lines

print("----------------------------------------------- Read/Clean Data -----------------------------------------------")

# Read dataset into pandas dataframe
dataset = pd.read_csv('data/books_1.csv', thousands=",", encoding="unicode_escape", error_bad_lines=False,
                      na_filter=False)

# Data exploration
print(dataset.describe())

# Clean data: replace null values with np.nan so rows with null data in columns needed for processing can be deleted
dataset.replace(["", "NaN", "NaT", "nan"], np.nan, inplace=True)

# Delete rows based on nan values in specified columns
dataset = dataset.dropna(subset=["author", "rate", "title", "num_of_page", "genre", "rating_count"], inplace=False)

# Regex to remove extraneous commas from column
dataset["publisher"] = dataset["publisher"].replace(r",", "", regex=True, inplace=True)
dataset.fillna("", inplace=True)
print(dataset.rating_count.isnull().any().any())

# Drop unnecessary columns
dataset = dataset.drop(["book_link"], 1, )
dataset = dataset.drop(["id"], 1)
dataset = dataset.drop(["series"], 1)

# Remove newline from cells
dataset = dataset.replace(r"\n", "", regex=True)

# Remove whitespace from column
dataset["publisher"] = dataset["publisher"].str.strip()

# Convert ratings strings to float and remove ratings below a certain threshold to keep only books with ratings of
# "average" or above
col = ["rate"]
dataset["rate"] = dataset[col].replace("", regex=True).astype(float)
low_ratings = dataset[dataset["rate"] < 2.6].index
dataset.drop(low_ratings, inplace=True)

# Remove all non-English books
dataset.drop(dataset[dataset["lang"] != "English"].index, inplace=True)

# Convert rating count column to int and eliminate outliers
dataset['rating_count'] = dataset['rating_count'].str.replace(',', '').astype(int)

# Convert count to string, replace commas, then convert back to int
dataset["rating_count"] = dataset["rating_count"].astype(str)
try:
    print(dataset.rating_count.isnull().any().any())
    dataset.rating_count = dataset.rating_count.apply(lambda x: x.replace(',', ''))
    dataset['rating_count'] = dataset['rating_count'].astype(np.int64)
    print(dataset.dtypes)
except:
    print("Having trouble converting")

# Remove duplicate rows
print(len(dataset))
dataset.drop_duplicates(keep="first")
print(f"Dataset size is now = " + str(len(dataset.index)))

# Ensure column has correct data type
dataset["rate"].astype(float)

print("--------------------------------------------- End Read/Clean Data ---------------------------------------------")

print("--------------------------------------- Data Exploration/Visualization ----------------------------------------")

# Data exploration
print(f"data.describe: \n{dataset.describe()}")


# Calculate ratings distribution
def rating_distribution():
    f = plt.figure(figsize=(7, 7))
    sns.kdeplot(dataset['rate'], shade=True)
    plt.title('Book Rating Distribution\n')
    plt.xlabel('\nRating')
    plt.ylabel('\nFrequency\n')
    plt.show()


# Splits genre into lists
dataset["genre"] = (dataset["genre"].str.split(", "))

# Sort by rating in descending order
dataset.sort_values(by=["rate"], ascending=False, ignore_index=False, inplace=True)
pd.set_option("display.max_columns", 30)


# Calculate/display books with the most ratings
def most_ratings():
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    popular_books = dataset.nlargest(10, ['rating_count']).set_index('title')['rating_count']
    ax = sns.barplot(popular_books, popular_books.index)
    ax.set(xlabel="\n\nRating Count (millions)", ylabel="Book Title\n\n")
    plt.tight_layout()
    plt.show()


def rating_by_rating_count():
    fig = plt.gcf
    ax = sns.relplot(x="rate", y="rating_count", data=dataset, color='red', sizes=(100, 200), height=7,
                     marker='o')
    ax.set_axis_labels("\nAverage Rating", "\nRating Count\n\n")
    sns.set(rc={"figure.figsize": (1, 1)})
    plt.tight_layout()
    plt.show()


# Draw plots
rating_distribution()
most_ratings()
rating_by_rating_count()

print("-------------------------------- Additional Data Cleaning and Normalization -----------------------------------")

# Convert review count column to int and eliminate outliers
dataset['review_count'] = dataset['review_count'].str.replace(',', '').astype(int)
dataset["review_count"] = dataset["review_count"].astype(int)
dataset = dataset[(np.abs(stats.zscore(dataset["review_count"])) < 3)]

# Calculate Z-score of book ratings, page count, review count, then remove outliers not within 3
# standard deviations from mean
dataset = dataset[(np.abs(stats.zscore(dataset["rate"])) < 3)]

# Convert page count column to int and eliminate outliers
dataset["num_of_page"] = dataset["num_of_page"].astype(int)
dataset = dataset[(np.abs(stats.zscore(dataset["num_of_page"])) < 3)]


print("------------------------------------- End Data Cleaning and Normalization -------------------------------------")

print("-------------------------------------------- Recommendation Engine --------------------------------------------")


# Create column to store combined features, then add to existing dataframe
def combine_cols(pd_frame):

    # Split list of lists in genre column
    pd_frame["genre_string"] = pd_frame['genre'].apply(lambda x: ','.join(map(str, x)))

    temp = pd.DataFrame(data=pd_frame, columns=["title", "genre_string", "author"])
    temp["combined"] = temp.values.tolist()
    pd_frame["new_Column"] = temp["combined"]
    pd.set_option("display.max_colwidth", None)
    return pd_frame


# Create new dataset with column containing combined features
dataset1 = combine_cols(dataset)
boolean = not dataset1["title"].is_unique
print(f"boolean: {boolean}")
dataset1 = dataset1.drop_duplicates(subset=["title"], keep="first")
print(f"boolean: {boolean}")


def transform_data(data_combine, data_plot):

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(data_combine['new_Column'].astype(str))
    cos_sim = cosine_similarity(count_matrix)

    print(f"Cos sim: {cos_sim}")

    return cos_sim


# Resetting book id's using current dataframe index values, then creating a new column of book id's which correspond
# to the new index values
dataset1.reset_index(inplace=True, drop=True)
# print(dataset1.head(2))
dex = dataset1.index.values
# print(dex[0:4])
dataset1.insert(0, column="book_id", value=dex)


def rec_engine(title, data, combine, transform):

    indices = pd.Series(data.index, index=dataset1["title"])
    index = indices[title]

    cs_scores = list(enumerate(transform[index]))
    cs_scores = sorted(cs_scores, key=lambda x: x[1], reverse=True)
    cs_scores = cs_scores[1:6]

    book_indices = [i[0] for i in cs_scores]

    title = dataset1["title"].iloc[book_indices]
    genres = dataset1["genre"].iloc[book_indices]
    rating = dataset1["rate"].iloc[book_indices]

    recommendation_data = pd.DataFrame(columns=["Title", "Genre", "Rating"])

    recommendation_data['Title'] = title
    recommendation_data['Genre'] = genres
    recommendation_data['Rating'] = rating

    return recommendation_data


def get_book_data():
    book_data = dataset1.drop_duplicates(subset=["title"], keep="first")
    return book_data


def get_matches(book_title):

    find_book = get_book_data()
    combine_result = combine_cols(find_book)
    transform_result = transform_data(combine_result, find_book)
    if book_title not in find_book["title"].unique():
        return "Sorry, we don't have that book yet."
    else:
        recommendations = rec_engine(book_title, find_book, combine_result, transform_result)
        return recommendations.to_dict("records")


def rating_by_rating_count1():
    ax = sns.relplot(x="rate", y="rating_count", data=dataset1, color='green', sizes=(100, 200), height=7,
                     marker='o')
    ax.set_axis_labels("\nAverage Rating", "\nReview Count\n\n")
    sns.set(rc={"figure.figsize": (1, 1)})
    plt.tight_layout()
    plt.show()


rating_by_rating_count1()


print("------------------------------------------ End Recommendation Engine ------------------------------------------")

print("---------------------------------------------- Linear Regression ----------------------------------------------")


# Testing linear regression with sample genre
dataset5 = dataset1
print(f"Shape: {dataset5.shape}")
dataset5 = dataset5[dataset5["genre_string"].str.contains("Fantasy")]
print(dataset5["genre_string"].head(5))
user_query = input("Please enter a genre: \n")

dataset5["genre_string"] = dataset5.genre_string.astype("category")
dataset5["genre_string"] = dataset5.genre_string.str.contains(user_query)
X = dataset5["genre_string"].values.reshape(-1, 1)
y = dataset5["rate"].values.reshape(-1, 1)

# X = pd.get_dummies(data=X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print("x train shape", X_train.shape)
# print(y_train.shape)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# print(regressor.intercept_)
# print(regressor.coef_)
y_pred = regressor.predict(X_test)
dataset3 = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
print(dataset3.head(20))
# dataset2 = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})

# Graphing first 20 to visualize actual vs predicted values
linreg_graph = dataset3.head(20)
linreg_graph.plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle="-", linewidth="0.5", color="black")
plt.show()


print(f"TEST GENRE: {dataset5.genre_string.head(5)}")


def algo_analytics():
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_num = float(rmse)
    algo_values = {"MAE": mae, "MSE": mse, "RMSE": rmse_num}
    x = list(algo_values.keys())
    y = list(algo_values.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='blue',
            width=0.5)

    print("Mean absolute error: ", mae)
    print("Mean squared error: ", mse)
    print("Root mean squared error: ", rmse)

    plt.show()


def cos_sim_graph():
    dataset6 = dataset1.drop(["book_id", "genre", "new_Column", "title", "lang", "date_published",
                              "publisher", "award", "genre_string", "author"], 1)

    # calculate the correlation matrix
    corr = dataset6.corr()

    # plot the heatmap
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.subplots_adjust(bottom=0.28)
    plt.show()


cos_sim_graph()
algo_analytics()


print("-------------------------------------------- End Linear Regression --------------------------------------------")

print("------------------------------------------------ Run Flask App ------------------------------------------------")

app = Flask(__name__, template_folder="templates")


@app.route("/recommend", methods=["POST", "GET"])
def recommend():
    book_1 = request.form.get("book")
    results = get_matches(book_1)
    parsed_results = jsonify(results)
    return render_template_string('''
        <table style="width: 75%; alignment: center; padding-left: 20px">
            <h1 style="padding-left: 20px; padding-top:25px">Your results are...</h1>
            <br>
            {% if results %}
             <tr>
               {% for key in results[0] %}
                    <th> {{ key }} </th>
                   {% endfor %}
             </tr>
                {% endif %}

               {% for dict_item in results %}
               <tr>
                   {% for value in dict_item.values() %}
                       <td style="background-color: antiquewhite"> {{ value }} </td>
                       {% endfor %}
               </tr>
                {% endfor %}
        </table>
        <br>
        <br>
        <br>
        <a style="padding-left: 20px" href="/">Back to home</a>
    ''', results=results)


@app.route("/", methods=["GET", "POST"])
def draw_graphs():
    logging.basicConfig(filename='logger.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s '
                                                                           '%(threadName)s : %(message)s')

    def log():
        app.logger.info('Information level log')
        app.logger.warning('Warning level log')
        return render_template("index.html")

    # Display ratings by page count
    # ----------------------------------------------------------------------------------------------------
    ratings = []
    pages = []

    for i in dataset["num_of_page"].head(100):
        pages.append(i)
    for j in dataset["rate"].head(100):
        ratings.append(j)

    labels = [row for row in ratings]
    values = [row for row in pages]

    # ----------------------------------------------------------------------------------------------------

    category = ["Fantasy", "Romance", "Mystery"]

    # Display ratings by genre
    dataset1a = dataset[dataset['genre'].str.join(' ').str.contains('Fantasy')]
    dataset1b = dataset[dataset['genre'].str.join(' ').str.contains('Romance')]
    dataset1c = dataset[dataset['genre'].str.join(' ').str.contains('Mystery')]

    ratings1 = []
    fantasy = []
    romance = []
    mystery = []

    for i in dataset1a.head(100):
        fantasy.append(i)
    for j in dataset1b.head(100):
        romance.append(j)
    for k in dataset1c.head(100):
        mystery.append(k)

    fantasy_mean = np.mean(dataset1a["rate"])
    ratings1.append(fantasy_mean)
    romance_mean = np.mean(dataset1b["rate"])
    ratings1.append(romance_mean)
    mystery_mean = np.mean(dataset1b["rate"])
    ratings1.append(mystery_mean)

    labels1 = [row for row in category]
    values1 = [row for row in ratings1]

    # ----------------------------------------------------------------------------------------------------

    # Compare ratings of fiction and nonfiction books
    fiction_ratings = []
    nonfiction_ratings = []
    f_nf_ratings = []
    category1 = ["Fiction", "Nonfiction"]

    dataset2 = dataset[dataset['genre'].str.join(' ').str.contains('Fiction')]
    dataset2a = dataset[dataset['genre'].str.join(' ').str.contains('Nonfiction')]

    for i in dataset2["rate"].head(100):
        fiction_ratings.append(i)
    for j in dataset2a["rate"].head(100):
        nonfiction_ratings.append(j)
    fiction_mean = np.mean(fiction_ratings)
    print(fiction_mean)
    nonfiction_mean = np.mean(nonfiction_ratings)
    print(nonfiction_mean)
    f_nf_ratings.append(fiction_mean)
    f_nf_ratings.append(nonfiction_mean)
    labels2 = category1
    values2 = [row for row in f_nf_ratings]

    log()

    return render_template("index.html", labels=labels, values=values, labels1=labels1, values1=values1,
                           labels2=labels2, values2=values2)


if __name__ == '__main__':
    app.run(debug=True)

# <---------------------------------------------------- SCRAP HEAP ---------------------------------------------------->

# Test plot
# labels = [row[0] for row in data]
# values = [row[1] for row in data]

# labels = ["01-01-2020", "02-01-2020", "03-01-2020", "04-01-2020", "05-01-2020", "06-01-2020"]
# values = [2343, 4353, 7343, 7311, 2342, 5684]
# for row in data:
# labels.append(row[0])
# values.append(row[1])
# df = dataset({
#     "genre",
#     "rate"
# })
#
# return df.to_html(header="true", table_id="table")

# return render_template('index.html', column_names=df.columns.values, row_data=list(df.values.tolist()))

# Printing first 26 rows to check data
# pd.set_option("display.max_columns", 30)
# print(dataset.iloc[:26, : 30])

# Classify variables
# dataset["rating by page count"] = pd.cut(dataset["rating by page count"], bins=[3.4, 4.0, 4.5, 5.0], labels=[0, 1])

# x = dataset[dataset.columns[:-1]]
# y = dataset["rating by page count"]
# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=.20, random_state=42)

# fig = px.line(dataset, x="num_of_page", y="rate", title="XXX")
# fig.update_xaxes(rangeslider_visible=True)
# fig.show()

# def plotly_pages_rating:
#     fig = px.line(dataset, x="num_of_page", y="rate", title="XXX")
#     fig = fig.update_xaxes(rangeslider_visible=True)
#     fig.update_layout (width=1500, height=500)
#     plot_json = json.dumps(fig, cls=plot.utils.PlotlyJSONEncoder)
#     return plot_json

# df = pd.DataFrame(dataset)
# print(dataset.columns)
# pd.set_option("display.max_columns", 30)
# dataset1 = dataset[dataset.num_of_page != "".strip]
# dataset = dataset.dropna(columns=["num_of_page"])
# dataset1 = df[df["num_of_page"].notna()]
# inplace=True

# print(f"Null Sum: " + str(dataset.isna().any().sum()))

# desired_width = 640
# desired_height = 200

# pd.set_option("display.width", desired_width)
# np.set_printoptions(linewidth=desired_width)

# print(dataset.head(5))
# print(dataset.tail(5))

# Genre distribution
# dataset["rating_count"].value_counts().plot(x="num_of_page", y="genre", kind="bar", figsize=(20, 10))
# plt.locator_params(axis="x", nbins=10)
# # dataset.reset_index().plot(kind="scatter", x="genre", y="rate")
#
# plt.show()

# https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js
# https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.esm.js
# https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.esm.min.js
# https://cdn.jsdelivr.net/npm/chart.js@2.9.4/dist/Chart.min.js

# ----------------------------------------------------------------------------------------------------

# Correlate data
# plot_cats = []

# f = plt.figure(figsize=(19, 15))
# plt.matshow(dataset.corr(), fignum=f.number)
# plt.xticks(range(dataset.select_dtypes(['number']).shape[1]), dataset.select_dtypes(['number']).columns,
#            fontsize=14,
#            rotation=45)
# plt.yticks(range(dataset.select_dtypes(['number']).shape[1]), dataset.select_dtypes(['number']).columns,
#            fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
# plt.show()
# print(dataset.corr())

# @app.route('/')
# def home():
#     if not session.get('logged_in'):
#     return render_template('login.html')
#     else:
#     return "Hello Boss!"

# Route for handling the login page logic
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if request.form['username'] != 'admin' or request.form['password'] != 'admin':
#             error = 'Invalid Credentials. Please try again.'
#         else:
#             return redirect(url_for('home'))
#     return render_template('login.html', error=error)


# print(pd_frame.genre_string.head(5))
# pd_frame['genre'] = [','.join(map(str, l)) for l in pd_frame['genre']]
# pd_frame['genre'] = pd_frame['genre'].agg(lambda x: ','.join(map(str, x)))
# pd_frame['genre'].apply(lambda x: ','.join(map(str, x)))
# pd_frame['genre'] = pd_frame.genre.apply(lambda x: ', '.join([str(i) for i in x]))
# pd_frame['genre'] = pd_frame['genre'].apply(lambda x: x[1:-1])

# Login credentials
# authorized_users = {"admin": "admin"}


# @app.route("/")
# def landing_page():
#     return render_template("login.html")
#
#
# # Simple login page
# @app.route("/form_login", methods=["POST", "GET"])
# def do_admin_login():
#     request_name = request.form["username"]
#     request_pw = request.form["password"]
#     if request_name not in authorized_users:
#         return render_template("login.html", info="Invalid username. Please check and try again.")
#     if request_pw not in authorized_users:
#         return render_template("login.html", info="Invalid password. Please check and try again.")
#     else:
#         if request.form["password"] == 'admin' and request.form['username'] == 'admin':
#             return render_template("index.html")
#     return render_template("index.html")


# def popular_authors():
#     plt.subplots(figsize=(10, 13))
#     plt.title('Most Popular Authors by Books Published\n\n', fontsize=16)
#     dataset.author.value_counts()[:10].plot(kind="bar")
#     plt.show()
#     # print(dataset.head(5))
#
#
# popular_authors()


# # test poinst for ratings_count=40, Avg_Rating= 4.5, Publish_year= 2002, Pages_no= 523
# test_point = [40, "Jim Butcher", "Fantasy", 523, 50]
# X = dataset.iloc[:, [2, 3, 4, 6, 7]].values
# # build a nearest neighbor object, we are searching for just 3 neighbors so n_neighbors=3
# nn = NearestNeighbors(n_neighbors=3).fit(X)
# # kneighbors returns the neighbor for the test_point
# print(nn.kneighbors([test_point]))


# # Calculate/display top ten
# def top_ten_books():
#     top_ten = dataset[dataset['rating_count'] > 1000000]
#     top_ten.sort_values(by='rate', ascending=False)
#     plt.style.use('seaborn-whitegrid')
#     plt.figure(figsize=(10, 10))
#     data = top_ten.sort_values(by='rate', ascending=False).head(10)
#     sns.barplot(x="rate", y="title", data=data, palette='inferno')
#     plt.show()
#
#
# top_ten_books()


# Combine columns needed for analysis/recommendation functions
# def combine_columns(data):
#     features = []
#     # corpus = dataset["genre"].tolist()
#     for i in range(0, data.shape[0]):
#         features.append(data["title"][i] + data["author"][i])
#     return features

# Create column to store combined features
# dataset["combined_dataset"] = combine_columns(dataset)


# try:
#     # Remove commas from column, convert bad data to NaN then remove
#     dataset['rating_count'] = pd.to_numeric(dataset['rating_count'], errors='coerce')
#     dataset = dataset.dropna('rating_count')
#     dataset['rating_count'] = dataset['rating_count'].astype(np.int64)
# except:
#     print("can't convert")
#     print(dataset.dtypes)

# def cs_graph():
#
#     # hist_data = dataset.copy()
#
# fig1 = plt.gcf() fig1.set_size_inches(12, 12) plt.matshow(cos_sim.corr(), fignum=fig1.number) plt.xticks(range(
# dataset.select_dtypes(['object', 'number']).shape[1]), dataset.select_dtypes(['object', 'number']).columns,
# fontsize=8, rotation=45) plt.yticks(range(dataset.select_dtypes(['object', 'number']).shape[1]),
# dataset.select_dtypes(['object', 'number']).columns, fontsize=8) cb = plt.colorbar() cb.ax.tick_params(
# labelsize=10) plt.title('Correlation Matrix\n\n', fontsize=10) plt.show()

# # Create loop to print the titles of the first five books from the sorted list
# print(dataset1.head(5))
# j = 0
# print("The 5 most recommended books to " + test_title + " are: \n")
#
# for item in top_five:
#     book_title = dataset1[dataset1["book_id"] == item[0]]["title"].values[0]
#     print(j + 1, book_title)
#     j = j + 1
#     if j >= 6:
#         print(book_title)
#         break


# @app.route("/search", methods=["GET", "POST"])
# def search():
#     form = SearchForm()
#     if form.validate_on_submit():
#         search_term = form.query.data
#         results = Foo.query.all()
#         return render_template('index.html', form=form, results=results)
#     return render_template('index.html', form=form)


# @app.route("/get_matches", methods=["GET"])
# def rec_books():
#     results = get_matches(request.args.get("title"))
#     return jsonify(results), render_template("index.html")

# @app.route("/rec_books", methods=["GET", "POST"])
# def rec_books():
#     results = get_matches(request.args.get("title"))
#     return results  # results, redirect(url_for("index"))  # jsonify(results), render_template("index.html")

# @app.route("/")
# def home():
#     return render_template("index.html")
#
#
# print(dataset1.genre_string.head(5))
# print(dataset.rate.isnull().any().any())
# dummy = pd.get_dummies(dataset1[["genre_string"]])

# TODO Leave this here for cosine similarity test purposes
# Testing to see if dataframe index matches random number
# print(dataset1["title"][5])
# test_title = dataset1["title"][5]
# print("Test title is: " + test_title)
#
# # Testing to make sure we can retrieve the book id
# book_id = dataset1[dataset1.title == test_title].index.values[0]
# print(f"Book ID: {book_id}")
#
# # Creating a list of tuples containing the book id and similiarity score for the test book
# scores = list(enumerate(cos_sim[book_id]))
# print(f"Scores are: {scores[0:5]}")
#
# # Sort list of similar books in descending order
# sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
# top_five = (sorted_scores[1:6])
# print(top_five)
# print(sorted_scores[1:5])


# def linreg_bar():
#     linreg_bar.plot(kind="bar", figsize=(16, 10))
#     plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
#     plt.grid(which="minor", linestyle="-", linewidth="0.5", color="black")
#     plt.show()

# def heatmap_corr():
#
#     dataset_heatmap = dataset.pivot
#     # sns.heatmap = dataset_heatmap
#
#     # calculate the correlation matrix
#     corr = dataset.corr()
#
#     # plot the heatmap
#     sns.heatmap(dataset_heatmap,
#                 xticklabels=corr.columns,
#                 yticklabels=corr.columns)

    # fig = plt.figure()
    # fig.set_size_inches(8, 8)
    # plt.matshow(dataset.corr(), fignum=fig.number)
    # plt.xticks(range(dataset.select_dtypes(['number', "object"]).shape[1]), dataset.select_dtypes(["number", 'object']).columns,
    #            fontsize=8,
    #            rotation=45)
    # plt.yticks(range(dataset.select_dtypes(['number', "object"]).shape[1]), dataset.select_dtypes(["number", 'object']).columns,
    #            fontsize=8)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=10)
    # plt.title('Correlation Matrix\n\n', fontsize=10)
    # plt.show()

    # TODO This is the OG linear regression that works
    # dataset1["genre_string"] = dataset1.genre_string.astype("category")
    # X = dataset1["genre_string"].values.reshape(-1, 1)
    # y = dataset1["rate"].values.reshape(-1, 1)


# @app.route("/recommend", methods=["POST", "GET"])
# def recommend():
#     book_1 = request.form.get("book")
#     results = get_matches(book_1)
#     parsed_results = jsonify(results)
#     return parsed_results, render_template("recs.html")


# print(get_matches("Proven Guilty (The Dresden Files, #8)"))
# print(get_matches("Harry"))

# THIS ONE WORKS
# print(dataset1[dataset1["title"].str.contains("harry", regex=False, case=False)].title.head(5))

# def linreg_scatter():
#     le = preprocessing.LabelEncoder()
#     x_encoded = le.fit(X_test)
#     plt.figure()
#     plt.figure(figsize=(8, 8))
#     plt.scatter(x_encoded, y_test, color="red")
#     plt.plot(x_encoded, y_pred, color="black", linewidth=2)
#     plt.show()
#
# linreg_scatter()

# def linreg_scatter():
#     plt.figure()
#     plt.figure(figsize=(8, 8))
#     plt.scatter(X_test, y_test, color="red")
#     plt.plot(X_test, y_pred, color="black", linewidth=2)
#     plt.show()


    # TODO This works, but trying something new above
    # # Transform combined column to a matrix of word counts, using stop words to omit articles and such
    # col_matrix = CountVectorizer(stop_words="english").fit_transform(dataset1["new_Column"].astype(str))
    #
    # # Cosine similarity analysis of dataset to generate similarity scores
    # cos_sim = cosine_similarity(col_matrix)


# print(dataset1[dataset1["title"].str.contains(test_title, regex=False, case=False)].title.head(5))
# print(dataset1.title.head(20))

    # book_title = book_title.lower()
    # book_title = dataset1[dataset1["title"].str.contains(book_title, regex=False, case=False)].title
    # user_input = dataset1[dataset1["title"].str.contains("harry", regex=False, case=False)].title.head(5)
    # book_title = test_title

