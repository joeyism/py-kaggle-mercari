from pyspark.sql import functions
from pyspark.ml.feature import Word2Vec, StopWordsRemover
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import udf
import re
import itertools


# Loading Data
sqlContext = SQLContext(sc)
df = sqlContext.read.load("data/train.tsv", format="com.databricks.spark.csv", header="true", inferSchema="true", delimiter='\t')


# Viewing
# mean price by item_condition
#df.groupBy("item_condition_id").agg({"price": "mean"}).withColumnRenamed("avg(price)","mean_price").orderBy("mean_price").collect()
#
#
## Get all unique brand names
#join_text = functions.udf(lambda x: " ".join(set(x)))
#df.groupBy("item_condition_id").agg(functions.collect_list("brand_name").alias("brand_name")).withColumn("brand_name", join_text("brand_name")).collect()
#
## Get number of unique brands per item_condition_id
#df.groupBy("item_condition_id").agg(functions.countDistinct("brand_name")).orderBy("item_condition_id").collect()
#
## no of unique brands total
#df.select("brand_name").distinct().count()


def getAnalogy(s, model):
    qry = model.transform(s[0]) - model.transform(s[1]) - model.transform(s[2])
    res = model.findSynonyms((-1)*qry,5) # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]

def clean_description(text):
    text = text.lower().strip()
    text = re.sub("[^0-9a-zA-Z ]", " ", text)
    text = re.sub("[  ]", " ",text)
    return text

def group_by_words(dataframe):
    return dataframe.rdd.map(lambda x: x[0]).flatMap(lambda x: x.split(" ")).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

clean_description_udf = udf(clean_description)

df = df.fillna({"item_description": ""})
df = df.fillna({"category_name": ""})
df = df.withColumn("item_description", clean_description_udf("item_description"))
df = df.withColumn("description_words", functions.split("item_description", " "))
df = df.withColumn("id", monotonically_increasing_id())

category_dict = dict(df.select("category_name").distinct().rdd.map(lambda r: r[0]).zipWithIndex().collect())
global_category_dict = sc.broadcast(category_dict)
category_udf = udf(lambda x: global_category_dict.value[x])
df = df.withColumn("category_id", category_udf("category_name"))

# dict(group_by_words(df.select("category_name")).collect())
# see most common words
# group_by_words(df.select("item_description")).take(10)

word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="description_words", outputCol="word2vec_results")
model = word2Vec.fit(df.select("description_words"))
result = model.transform(df.select("description_words"))
result = result.withColumn("id", monotonically_increasing_id())
result = result.withColumn("word2vec_array", udf(lambda x: x.toArray())("word2vec_results"))

df = df.join(result, df["id"] == result["id"], "inner")
df = df.select(["item_condition_id", "price", "shipping", "word2vec_results"])
