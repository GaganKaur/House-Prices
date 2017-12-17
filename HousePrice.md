import pymysql
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


HOST = 'localhost'
PORT = 3306
USER = 'root'
PASSWD = 'W@ris0312'
DB = 'final_project'
TABLE_NAME = "housing_prices"
THRESHOLD = 0.7

FIELDS_WITH_NA = [	"LotShape","LandContour","Utilities",
					"LotConfig","LandSlope","BldgType",
					"HouseStyle","RoofStyle","RoofMatl","ExterQual","ExterCond",
					"Foundation","Heating","CentralAir",
					"KitchenQual","GarageType","GarageFinish","GarageCond",
					"PavedDrive","SaleType","Electrical",
				 ]

FEATURES_TO_NORMALIZE = [
							"LotArea","TotalBsmtSF",
							"1stFlrSF","2ndFlrSF",
							"GarageArea","PoolArea",
							"BsmtFullBath","BsmtHalfBath",
							"Fireplaces","GarageCars"
						]

FEATURES_NOT_TO_NORMALIZE = [
								"BsmtFullBath","BsmtHalfBath",
								"Fireplaces","GarageCars",
								"OverallQual","OverallCond",
								"BsmtQual","BsmtCond"
							]
FEATURES_TO_FLAT = [
						"LotShape","LandContour","Utilities",
						"LotConfig","LandSlope","BldgType",
						"HouseStyle","RoofStyle","RoofMatl","ExterQual","ExterCond",
						"Foundation","BsmtQual","BsmtCond","Heating","CentralAir",
						"KitchenQual","GarageType","GarageFinish","GarageCond",
						"PavedDrive","SaleType","Electrical",
						"YearBuilt","YearRemodAdd",
						"MoSold","YrSold"
					]

FEATURE_SET = [
				"LotShape",
				"LandContour",
				"Utilities",
				"LotConfig",
				"LandSlope",
				"BldgType",
				"HouseStyle",
				"RoofStyle",
				"RoofMatl",
				"ExterQual",
				"ExterCond",
				"Foundation",
				"BsmtQual",
				"BsmtCond",
				"Heating",
				"CentralAir",
				"KitchenQual",
				"GarageType",
				"GarageFinish",
				"GarageCond",
				"PavedDrive",
				"SaleType",
				"SaleCondition",
				"Electrical",
				"OverallQual",
				"OverallCond",
				"YearBuilt",
				"YearRemodAdd",
				"MoSold",
				"YrSold",
				"SalePrice"
			   ]

def creat_connection():
	conn = pymysql.connect(host=HOST, port=PORT, user=USER, passwd=PASSWD, db=DB)
	return conn

def sale_prices_plot_vs_feature(feature, db_connection):
	df = pd.read_sql("SELECT " + feature + ", avg(SalePrice) as Avg_Sell_Price FROM " + TABLE_NAME + " WHERE " + feature +" != 'NA' GROUP BY " + feature, con=db_connection)
	# plt.plot(df[feature], df["Avg_Sell_Price"])
	# print(type(df[feature]))
	x_ticks = list(df[feature])
	ind = np.arange(len(x_ticks))
	df.plot(kind="bar", xticks=ind, figsize=(10,10))
	plt.xticks(ind, x_ticks)
	plt.xlabel(feature)
	plt.ylabel("Average Selling Price")
	plt.show()

def flatten_feature(df):
	return pd.get_dummies(df).rename(columns = "{}_binary".format)

def normalize_feature(df, feature):
	x = df[feature].values.astype(float)
	x = x.reshape(-1,1)
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df_normalized = pd.DataFrame(x_scaled)
	return df_normalized

def build_test_train_dataset(df):
	msk = np.random.rand(len(df)) < THRESHOLD
	train = df[msk]
	test = df[~msk]
	return train, test

def build_regression_model(X_train, y_train):
	print(X_train.head())
	params = {'n_estimators': 1000, 'max_depth': 15, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
	regr = GradientBoostingRegressor(**params)
	regr.fit(X_train, y_train)
	return regr

def predict_test_data(X_test, regr):
	y_test = regr.predict(X_test)
	return y_test

def main():
	db_connection = creat_connection()
	df_with_normalizing_features = pd.read_sql("SELECT " + ",".join(FEATURES_TO_NORMALIZE) + " FROM " + TABLE_NAME, con=db_connection)
	df_with_categorical_features = pd.read_sql("SELECT " + ",".join(FEATURES_TO_FLAT) + " FROM " + TABLE_NAME, con=db_connection)
	df_complete = pd.read_sql("SELECT " + ",".join(FEATURE_SET) + " FROM " + TABLE_NAME, con=db_connection)
	# Replace NA values with mode
	for f in FIELDS_WITH_NA:
		mode = df_complete[f].mode().iloc[0]
		x = df_complete[f].str.replace('NA', mode)
		df_complete = df_complete.drop(f, axis=1)
		df_with_categorical_features = df_with_categorical_features.drop(f, axis=1)
		df_complete[f] = x
		df_with_categorical_features[f] = x
	# plotting selling prices vs various features
	for f in FEATURE_SET:
		sale_prices_plot_vs_feature(f, db_connection)
	# Normalizing the features between 0 and 1
	for f in FEATURES_TO_NORMALIZE:
		if f not in FEATURES_NOT_TO_NORMALIZE:
			x_norm = normalize_feature(df_with_normalizing_features, f)
			df_with_normalizing_features = df_with_normalizing_features.drop(f, axis=1)
			df_with_normalizing_features[f] = x_norm
	for f in FEATURES_TO_FLAT:
		x_flat = flatten_feature(df_with_categorical_features[f])
		df_with_categorical_features = df_with_categorical_features.drop(f, axis=1)
		df_with_categorical_features = pd.concat([df_with_categorical_features, x_flat], axis=1)
	final_data_set = pd.concat([df_with_normalizing_features, df_with_categorical_features], axis=1)
	final_data_set['SalePrice'] = df_complete['SalePrice']
	X_train, X_test = build_test_train_dataset(final_data_set)
	y_train = X_train['SalePrice']
	X_train = X_train.drop('SalePrice', axis=1)
	y_test = X_test['SalePrice']
	X_test = X_test.drop('SalePrice', axis=1)
	print("X_train_shape: " + str(X_train.shape))
	print("X_test_shape: " + str(X_test.shape))
	print("y_train_shape: " + str(y_train.shape))
	print("y_test_shape: " + str(y_test.shape))
	regression_model = build_regression_model(X_train, y_train)
	y_pred = predict_test_data(X_test, regression_model)
	print("Mean squared error: %.6f" % mean_squared_error(np.log(y_test), np.log(y_pred)))

if __name__ == '__main__':
	main()