using DataFrames, MLJ, XGBoost


df = DataFrame(
  x1=rand(1000),
  x2=randn(1000),
  x3=rand(1000),
  x4=rand(1000),
  y=rand(1000)
)

df.x2 .= [x < 0 ? "Negative" : "Positive" for x in df.x2]

MLJ.schema(df)

coerce!(
  df,
  :x2 => Multiclass # OrderedFactor, Count
)

y, X = unpack(df, (name -> name == :y), (name -> true))

XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost



ContEnc = @load ContinuousEncoder pkg = MLJModels

ce_mod = ContEnc(drop_last=true)

ce_mach = machine(ce_mod, X)

fit!(ce_mach)

Xc = MLJ.transform(ce_mach, X)



model = XGBoostRegressor()

mach = machine(model, Xc, y)

fit!(mach);

yhat = MLJ.predict(mach, Xc)


RFR = @load RandomForestRegressor pkg = DecisionTree

forest = RFR()

r1 = MLJ.range(forest, :n_subfeatures, lower=1, upper=3)
r2 = MLJ.range(forest, :n_trees, lower=100, upper=1000)


tuned_forest_model = MLJ.TunedModel(
  model=forest,
  tuning=MLJ.Grid(resolution=10),
  resampling=CV(nfolds=10),
  ranges=[r1, r2],
  measure=MLJ.RootMeanSquaredError()
)

tuned_machine = machine(tuned_forest_model, Xc, y)

fit!(tuned_machine)

yhat = MLJ.predict(tuned_machine, Xc)

MLJ.rms(yhat, y)

train_idx, test_idx = MLJ.partition(eachindex(y), 0.7)


#====


=====#

mod1 = XGBoostRegressor()
mach1 = machine(mod1, Xc, y)
fit!(mach1, rows=train_idx)

yhat1_test = MLJ.predict(mach1, Xc[test_idx, :])
y_test = y[test_idx]

MLJ.rmse(yhat1_test, y_test)

mach2 = machine(tuned_forest_model, Xc, y)

fit!(
  mach2,
  rows=train_idx
)

yhat2_test = MLJ.predict(mach2, Xc[test_idx, :])


MLJ.rmse(yhat2_test, y_test)