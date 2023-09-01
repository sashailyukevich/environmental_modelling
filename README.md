For this study, all modelling, calculation, and visualisation assessments are conducted with Python. 

First, referring to the official documentation, MAGICC (2023), and SALib SA are installed (Herman and Usher, 2017). 

Secondly, Python’s Data Analytics libraries Pandas, NumPy, Matplotlib, and Seaborn are imported (Harrison and Petrou, 2020). Additionally, the following MAGICC’s and SALib’s libraries are used:

1.	Pymagicc model evaluation environment.
2.	Scmdata for handling SCM.
3.	Read_scen_file for inputting a bespoke scenario.
4.	RCP 2.6 for running the RCP 2.6 climate change pathway.
5.	Saltelli parameter sample generator.
6.	PAWN Global SA method.

Thirdly, the input data, specific to this project’s evaluations, are listed as the following variables:

1.	The strings OBSERVED_DATA_FILEPATH and
SIMULATED_DATA_FILEPATH for reading CSV files with the defined file pathways.
2.	The lists OBSERVED_DATA_COLS_LIST 
and SIMULATED_DATA_COLS_LIST for selecting only relevant CSV files’ columns for this study.
3.	The list PARAMETERS for specifying parameters’ perturbation in the input.
4.	The integers MINIMUM_YEAR and MAXIMUM_YEAR for specifying the period for model’s future projection.
5.	The dictionary PROBLEM for inputting PPE.
6.	The float RATIO_BETWEEN_MODEL_AND_OBSERVED_ERRORS for selecting the model error ratio in relation to the observed error.
7.	The NumPy array OUTPUTS for generating SA indices.

Fourthly, 10 functions are designed as methods to run the project evaluations in sequential order. The methods are split into 3 categories:

1.	Parameters’ simulation, model run, and sensitivity analysis.

i.	Method 1: generate_model_inputs() generates parameter simulations with the Saltelli method as a NumPy array.
ii.	Method 2: run_model() returns temperature projections under RCP 2.6 with the given PPE as NumPy array and Pandas data frame.
iii.	Method 3: calculate_sa_indices() returns the PAWN sensitivity analysis indices as a data frame.
iv.	Method 4: visualise_sa_results() with Seaborn tools creates bar charts of the PAWN SA indices.

2.	Calibration.

i.	Method 5: get_and_prepare_data_for_calibration() read the CSV file as a data frame and renames columns.
ii.	Method 6: calculate_total_model_error() calculates a total model error for running the History Matching method.
iii.	Method 7: calibrate_data() executes calibration and returns a data frame with applicable columns. 
iv.	Method 8: evaluate_parameters_uncertainties_with_boxplots() with Seaborn tools creates boxplots to indicate parameters’ statistics and uncertainty after calibration.

3.	Model output projection.

i.	Method 9: get_post_calibration_projections_for_selected_period() with the merge function returns a data frame with calibrated parameters’ values for future projections,  and selects a specific period for those projections.
ii.	Method 10: visualise_calibrated_data_projections_with_lineplot() 
with Seaborn tools creates a line plot of model projections and their uncertainty band.

Additionally, numerous inquiries are run at the cleaning and exploration stages of the Data Analysis Pipeline, including the following Pandas functions:

1.	head() and tail() return the first and last n rows of a data frame respectively.
2.	info() informs about a data frame, including its data types, columns, null values, memory usage, and index.
3.	describe() delivers descriptive statistics of data.
4.	.shape indicates the number of rows and columns of a data frame.
5.	.columns shows column names
6.	.isna().sum() calculates the total number of NA values.

To complete, the data exploration stage indicated no missing data or any anomalies, which could have compromised further assessment.
# machine_learning
