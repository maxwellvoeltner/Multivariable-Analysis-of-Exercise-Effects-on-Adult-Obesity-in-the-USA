#!/usr/bin/env python
# coding: utf-8

# # Section 0: grading
# 
#   - *If you currently have a grade in the course that you are satisfied with:*
#     - **You do not need to complete this Final!** 
#     - The grade indicated on your Challenge Feedback sheet (on ELMS under 
#       Challenge 5 feedback) will be your final course grade.
#   
#   - *If you currently have less than a B in the course and you want to get a B:*
#     - Complete Sections 1 and 2, including all items where it says 
# 	  *COMPLETE THE FOLLOWING* or *ANSWER THE FOLLOWING*.
#     - If you are missing any previous Challenges, complete the items that 
#       indicate something like "if you are missing Challenge 1".
# 	
#   - *If you are going for an A in the course, you will need to:*
#     - Complete all three Sections, including all items where it says 
# 	  *COMPLETE THE FOLLOWING* or *ANSWER THE FOLLOWING*.
#     - If you are missing any previous Challenges, complete the items that 
#       indicate something like "if you are missing Challenge [X]".
#     - Decide if you want to do a Multiple Regression (multiple predictors), 
#       or a Logistic Regression (a binary categorical response variable).  
#       Complete the items indicated for those, in addition to other items.
# 
#   - *When completing sections that are for missing Challenges:*
#     - The code provided should help you complete it.
#     - The instructions and Readiness from those Challenges may contain
#       additional help and explanation if you need it.
# 
#   - **IF YOU INTEND TO COMPLETE A LOGISTIC REGRESSION**:
# 	- Use the `spotify-2023_CLEAN.csv` data set. The other data sets may not have
#       appropriate data for logistic regression.
#     - Pick either the `solo` or `major` variables to be your response variable 
#       (aka dependent variable, variable you are trying to predict).
#     - The `solo` variable is coded 1 if the song artist is a solo artist, 
#       and 0 if there is more than one person in the group.
#     - The `major` variable is coded 1 if the song is in a major key, and 0 if not.
# 
#   - **GENERAL TIP**:
#     - The data sets were all selected because there was "something to find" in
#       all of them. You might want to try a few different predictors, etc. before
#       settling on one. However, you are NOT required to find anything 
#       "statistically significant."
#     - All you need to do is to develop reasonable questions about which variable or
#       variables might predict another variable in the data, follow the procedure
#       below to test that hypothesis, and discuss the finding, whether or not you
#       found anything significant.  It's the process that matters!
#     - Good luck!
# 

# 
# # SECTION 1: exploratory analysis
# 
# Make sure to run the code to import modules first:

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf


# ## 1. Load Data
# 
# *COMPLETE THE FOLLOWING*
#   - Indicate which data set you are using for the Final.
#   - Load the data into R.
#   - See the code chunk below as an example.
#   - Throughout the examples, change YOURDATA to a variable of your choice.
#   - For example, you could pick "airlines" or "delays" if you are using
#     the Airline Delays data.
# 
# **IF YOU ARE USING THE SPOTIFY DATA**
#   - Use the second line in the chunk below, because the Spotify data
#     contains characters which don't work with the default UTF-8 encoding.

# In[2]:


# Nutrition Stats Dataset
obesity_data = pd.read_csv("Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System_OVERALL.csv")
# YOURDATA = pd.read_csv("data_sets/spotify-2023_CLEAN.csv", encoding = "ISO-8859-1")


# 
#   - Check the top few rows and the column names to make sure the data was 
#     read in correctly. (Example code below.)
# 

# In[3]:


print(obesity_data.head())


# In[4]:


print(obesity_data.columns)


# 
# ## 2. Describe variables of interest
# 
# *ANSWER THE FOLLOWING*
#   - Which variables are you using in your analysis?
#   - Describe each of them informally:
#     - Include the name of the variable and what it represents.
#   - Which variable is your **response** variable, (aka dependent variable or 
#     outcome variable) that you are trying to predict?
#   - Which variable(s) is/are your **predictor** variables (aka independent
#     variable), which you are using to predict the response?
# 

# Predictor Variables:
#     Q046 - Percent of adults who achieve at least 300+ minutes a week of moderate-intensity aerobic physical
#     activity or 150 minutes a week of vigorous-intensity aerobic activity (or an equivalent combination)
#     Q043 - Percent of adults who engage in muscle-strengthening activities on 2 or more days a week
# Response Variable:
#     Q036 - Percent of adults aged 18 years and older who have obesity

#   - **IF YOU ARE DOING MULTIPLE REGRESSION**: 
#     - Make sure you pick at least two predictors.
# 

#   - **IF YOU ARE DOING LOGISTIC REGRESSION**: 
#     - See note above about picking a binary response variable in the Spotify data.
# 

#   - **IF YOU ARE MISSING CHALLENGE 1**: 
#     - Make an educated guess or describe your initial expectations about 
#       possible values for the variables you selected above. What's the general 
#       range you'd expect? What kinds of values (numbers? large numbers or small? 
#       discrete integers or continuous (decimal) values?)

#   
# ## 3. Articulate research questions
# 
# *ANSWER THE FOLLOWING*:
#   - Why is it reasonable to expect that your predictor(s) might predict your 
#     response variable?
#   - Are you expecting positive or negative relationships? Why?
#     - Tip: a negative relationship in a regression means that (a) the parameter 
#       estimate is negative, and (b) as the predictor values get higher, the 
#       response value gets lower.
#   - If you find a positive or negative relationship, what would that mean? 
#     Why would it be interesting? (as in, real world impact)
#   - For example, who might it matter for, if you find a relationship between
#     variables?
# 

# The Q046 variable represents the percent of adults who perform sufficient exercise for maintaining a healthy life. Exercise burns calories (energy). The more exercise someone does, the less energy will be converted to fat (accumulation of fat leads to obesity). Therefore, it's reasonable to assume exercise amount has an effect on obesity.
# 
# The Q043Q represents the percent of adults who do muscle-strengthening activities twice or more per week.
# 
# The expectation is a negative relationship for Q046. As the percent of adults who achieve at least 300 minutes a week of moderate-intensity aerobic physical activity etc... goes up, the percentage of adults with obesity should go down because exercize combats obesity by using up energy (either calories or exisiting fat) rather than allowing energy to accumulate as fat (leading to obesity).
# 
# The expectation is negative for Q043. Muscle strengthening is a form of exercise, so it should burn calories and fat. Also, people who are strengthening their muscles are building muscle, and muscle burns more calories than fat, so people who have more muscle will store less energy and accumulate less fat.
# 
# If the relationship is negative, that would be expected. It would not be interesting because it would validate the concensus of the medical community that exercise is an effective way to avoid obesity.
# 
# If the relationship is positive, that would be suprising. That result would contradict the oncensus of the medical community that exercise is an effective way to avoid obesity.
# 
# The relationship results would matter for people concerned about obesity.

# ## 4. Univariate plots and details
# 
# *COMPLETE THE FOLLOWING*
#   - Create a histogram or density plot for each of your variables of interest, 
#     both predictor(s) and response variables.
#   - The following code plots a histogram of a variable called VAR from a data 
#     set called YOURDATA:
# 

# In[5]:


obesity_data.head()


# In[6]:


# predictor
sns.histplot(x = "Q046", data=obesity_data)


# This distribution is not normal. The drop off between the densities from 30-33 to 34-35 is too extreme to fit into the normal distribution categorization.

# 
#   - The following does the same, but with a density plot:
# 

# In[7]:


sns.kdeplot(x="Q046", data=obesity_data)


# This distribution does look more normal. It looks like it fits into the normal distribution categorization.

# In[8]:


# predictor
sns.histplot(x = "Q043", data=obesity_data)


# This distribution looks normal.

# In[9]:


sns.kdeplot(x="Q046", data=obesity_data)


# This distribution looks normal.

# In[10]:


# response
sns.histplot(x = "Q036", data=obesity_data)


# This distribution is not normal. The densities in the middle are not high enough to fit into the normal distribution categorization.

# In[11]:


sns.kdeplot(x="Q036", data=obesity_data)


# While this distribution is more normal looking, but still not normal. There are too many Q036 values with similar densities in the middle that make the plot too wide to fir the normal categorization.

# 
# *COMPLETE THE FOLLOWING*
#   - For each plot, comment on whether the variable looks like it is distributed 
#     roughly like a normal distribution.
#   - Comment on whether any variables seem like a good candidate for 
#     log-transformation, and if so, plot it both transformed and untransformed.
#   - You should use a histogram or barplot if you are examining a categorical
#     variable.
# 
# **IF YOU ARE MISSING CHALLENGE 2**:
#   - Examine (print) the minimum, maximum, mean, quartiles, and standard 
#     deviation of each variable.
#   - Plot a theoretical normal density with the same mean and standard deviation 
#     superimposed on a density plot of one variable.
# 
# To get all of these values at once, you can use the following (changing the name 
# of the data frame to match yours):

# In[12]:


obesity_data.describe()
# or get a subset of columns (change names to match columns you want)
# YOURDATA[["VAR2", "VAR2"]].describe()


# To generate theoretical densities from a normal distribution, you will need the 
# mean and standard deviation of the variable of interest.  The following works 
# for a variable called VAR for a data frame YOURDATA:

# In[13]:


VAR_values = obesity_data["Q046"]
x_grid = np.linspace(VAR_values.min(), VAR_values.max())
normal_densities = stats.norm.pdf(x_grid, loc = VAR_values.mean(), scale = np.std(VAR_values))


# 
# Then the following plots these normal densities as an orange curve, along with the 
# (empirical) density plot (in blue) of the variable VAR in data frame YOURDATA.
# 

# In[14]:


sns.kdeplot(x="Q046", data=obesity_data)
sns.lineplot(x=x_grid, y=normal_densities)


# 
# **IF YOU ARE MISSING CHALLENGE 2**
#   - Comment on whether the empirical density looks similar to the theoretical 
#     normal distribution.
# 

# The theoretical distribution for the predictor variable is similar to to the actual distribution.

# 
# **IF YOU ARE MISSING CHALLENGE 1**:
#   - Now that you have examined these variables, how do they match the 
#     expectations you described above?  Was there anything unexpected?
# 

# ## 5. Multivariate plots
# 
# *COMPLETE THE FOLLOWING*
#   - Create a scatter plot with the predictor on the x-axis and the response on 
#     the y-axis.
#   
# **IF YOU ARE MISSING CHALLENGE 4**:
#   - Create scatter plots with and without log transformations of both response 
#     and predictor variables (unless a variable is categorical).
# 	
# **IF YOU ARE DOING MULTIPLE REGRESSION**:
#   - Make scatter plots between each predictor and the response, separately.
#   - Also make a scatter plot between two of your *predictors*.
#   - Do they appear to be related? If so, is that potentially a problem?
#   
# **IF YOU ARE DOING LOGISTIC REGRESSION**:
#   - Instead of scatter plots with your (binary) response, plot density plots 
#     of your predictor(s), using color/fill to split the densities according to 
#     the response.

# 	
# The following code creates a scatter plot for predictor PRED and response RESP 
# in data frame YOURDATA:
# 

# In[15]:


sns.scatterplot(x = "Q046", y = "Q036", data=obesity_data)


# In[16]:


sns.scatterplot(x = "Q043", y = "Q036", data=obesity_data)


# In[17]:


sns.scatterplot(x = "Q046", y = "Q043", data=obesity_data)


# The 2 predictor variables do seem to be related. They are showing a strong positive correlation. This shouldn't be a problem because both vpredictors represent exercise. It is good that they have a strong positive relationship because it means they should both effect the reponse similarly. This will be helpful in determining the effects of the variables, because they should both affect the data similarly, so whether they do or do not will be useful in the analysis.

# 
# The following creates the split density plot if the RESPONSE variable is binary 
# categorical:
# 

# In[18]:


sns.kdeplot(x="Q046", data=obesity_data, hue="Q036")


# ## 6. Transformations for model-fitting
# 
# *COMPLETE THE FOLLOWING*
#   - After having plotted variables and some relationships, are there any 
#     transformations you think you should perform for the analysis? Why or why not?
#   - If you decide to transform a variable, create a new column with the
#     transformed value, to make it easier to plot and include in your model.
# 

# No. While the distributions aren't all normal, they are close enough to where it is appropriate to keep the data on the same scale that is is now. The log transformations just mess the distributions up.

# 
# **IF YOU ARE DOING LOGISTIC REGRESSION**:
#   - If you are just using a single predictor, you should *center* that predictor.
#   - See discussion below on what centering is and how to do it.
#   - If you have more than one predictor, read the discussion below for Multiple
#     Regression, and decide whether to center or standardize your predictors.
#   - Do *not* transform your response variable, because it needs to stay 0s and 1s
#     for a logistic regression.
# 

# 
# **IF YOU ARE DOING MULTIPLE REGRESSION**:
#   - Think about how you want to interpret the estimates of the effects of 
#     predictors with respect to how they relate to the response variable. More
#     specifically, do you want to:
#     1. interpret an effect as "a change of one [unit] of the predictor corresponds 
#        to a change in [parameter value] units of the response variable", OR
#     2. interpret an effect as "a change in one standard deviation of the 
#        predictor corresponds to a change in [parameter value] units of the response"?
# 
#   - For example, if you had an effect where a change in weight (predictor, where
#     the unit was pounds) corresponded to an increase in price (response), you 
#     could either say:
#     1. a change in X pounds corresponds to a change in Y dollars, OR
#     2. a change in 1 standard deviation from the average weight corresponds
#        to a change in Y dollars
# 	   
#   - The difference between these is whether you would like to talk about the 
#     predictors in absolute terms (the first one) or relative terms (the second one).
# 	
#   - If you want to talk in absolute terms, you should **center** your predictors.
#   - If you want to talk in relative terms, you should convert the predictors to 
#     **z-scores**.
#   - See below for explanations and how to do both of these in code.
# 

#   
# A "centered" variable is a variable where the mean is zero. The reason to do 
# this in a multiple regression is to make it easier to interpret the effects of 
# other predictors. Estimates of parameters, including the intercept and any slope 
# parameters, are all interpreted as "the estimate where other parameters are 
# zero." It's often the case that this doesn't make much sense with uncentered 
# variables.
# 
# For example, if you had a model predicting the price of diamonds (in dollars) 
# by weight (in carats) and depth (in millimeters), but weight and depth were 
# uncentered, then the intercept would represent the average price for diamonds 
# with a weight and depth of zero, but having a diamond with no weight and no 
# depth doesn't make sense. Furthermore, you would interpret the effect of weight 
# as the effect of a change in carats where depth was zero millimeters, which also 
# doesn't make much sense. If weight and depth were centered, then the intercept 
# would be interpreted as "the average diamond price for a diamond with an average 
# weight and an average depth", in other words, the average price of an average 
# diamond, which is a lot easier to interpret.  This is because centering changes 
# what the "zero" point means, because a value of zero for a centered weight 
# variable means "the average weight", not "a weight of zero carats".
# 
# Converting a variable to a *z-score* is just one further step, where you take 
# the centered variable and you divide it by the standard deviation of that 
# variable. What this does is put the variable on a "relative" scale.  So instead 
# of a change in 1 value meaning a change in one unit (like carats or millimeters), 
# it represents a change in 1 standard deviation.  This can be helpful if you 
# would rather be able to say things like "an unusually heavy diamond (change in 
# 2 standard deviations) would be much more valuable, all else equal, than an 
# unusually deep diamond."  In other words, it's helpful to convert predictors to 
# z-scores (also called *standardizing* variables) if you want to put variables on 
# a common scale so that you can compare their effects, like saying that weight 
# has a larger effect than depth. Note that z-scores are also centered, so all of 
# the interpretation of centering also applies.
# 
# It is up to you whether you center or standardize (convert to z-scores) your 
# predictors, but for this assignment, you should do one or the other (if you are 
# doing multiple regression).
# 
# To center a predictor VAR, you should create a new column that represent the 
# variable minus its mean. Make sure to do this for **each** predictor (but not 
# the response variable). I recommend adding a lowercase "c" (for "centered") to 
# the variable name to represent your centered variable, or a "z" to represent a 
# standardized (z-score) variable. It helps you remember which variable it came 
# from originally, and it helps remind you that it's centered/standardized.
# 

# In[19]:


# obesity_data["cQ046"] = YOURDATA["Q046"] - YOURDATA["Q046"].mean()
# YOURDATA["cQ043"] = YOURDATA["Q043"] - YOURDATA["Q043"].mean()


# 
# To convert a predictor to z-scores, simply divide the centered variable by the 
# standard deviation:
# 

# In[20]:


obesity_data["cQ046"] = (obesity_data["Q046"] - obesity_data["Q046"].mean())/obesity_data["Q046"].std()
obesity_data["cQ043"] = (obesity_data["Q043"] - obesity_data["Q043"].mean())/obesity_data["Q043"].std()


# # Section 2: Model Fitting

# ## 1. Fit your model
# 
# *COMPLETE THE FOLLOWING*
#   - Use the `smf.ols()` function and the `fit()` method to fit your model (unless you are performing 
#     logistic regression).
#   - Include the intercept by representing it as a "1" in the model.
#   
# To fit a model called MODEL with two predictors zPRED1 and zPRED2 predicting a 
# response RESPONSE in data frame YOURDATA:
# 
# (As throughout, alter the code to fit the names of your variables.)

# In[21]:


MODEL = smf.ols("Q036 ~ 1 + cQ046 + cQ043", data = obesity_data).fit()


# **IF YOU ARE PERFORMING LOGISTIC REGRESSION**:
#   - Make sure your response variable is coded as 0s and 1s.
#   - Use the `smf.logit()` function with the `fit()` method:

# In[22]:


# LOGISTIC_MODEL = smf.logit("Q036 ~ 1 + Q046 + Q043", data = YOURDATA).fit()


# ## 2. Report the estimates and standard errors
# 
# *COMPLETE THE FOLLOWING*:
#   - Print out the estimates and standard errors of the model.
#   - Describe the slope effect(s) (not the Intercept):
#     - Are the effects positive or negative?
#     - Which slope seems to be the largest (in magnitude, regardless of 
#       positive/negative)?
#     - Do any effects seem large compared to their standard errors? 
#       (i.e., at least 2 times the size of their standard error)
# 
# To see the estimates (parameters) and standard errors for a model called MODEL:
# 

# In[23]:


print(MODEL.params)
print(MODEL.bse)


# 
# **IF YOU ARE MISSING CHALLENGE 5**:
#   - Pick one predictor, and plot a scatter plot of it on the x-axis with the 
#     response on the y-axis.
#   - Now use the values you obtained from the estimates above to add a blue 
#     regression line to the plot.
#   - Make sure you use the exact predictor variable from your model to plot 
#     the scatter plot (for example, the z-score version if that's what's in 
#     your model)
#   
# To add a line with intercept of INT and slope of SLOPE to a scatter plot of 
# a predictor zPRED1 and response RESPONSE:
# 

# In[24]:


sns.scatterplot(x="cQ046", y="Q036", data=obesity_data)
plt.axline(xy1=(0, 28.528481), slope=-1.005310)
plt.show()


# 
# # STOP HERE IF YOU ARE ONLY TRYING TO GET TO A GRADE OF *B* IN THE COURSE
# 
#   - Section 3 is for people trying to get an A.
# 

# 
# # SECTION 3: Inference and Interpretation
# 

# 
# ## 1. Examine residual plots
# 
#   - In order to examine how well the model is fitting, it is a good idea to 
#     examine the model residuals.
#   - Two good visualizations are:
#     1. A histogram or density plot of the residuals.
#     2. A scatter plot with model predictions on the x-axis and residuals on the y-axis.
# 	
#   - The histogram of residuals should ideally look roughly like a normal
#     distribution. If it looks very non-normal, you might want to reconsider 
#     transformations, or it might indicate some other problem with the model.
#   - The scatter plot should look largely random. If there are any strong 
#     patterns,  it might again suggest that the model is making systematic errors
#     of one kind of another.
#   
# In order to plot these values, it's easiest if they are first added as new 
# columns to the data frame. To create a column of model residuals from a model 
# called MODEL for a data frame YOURDATA:
# 

# In[25]:


obesity_data["residuals"] = MODEL.resid


# 
# To create a column of model predictions:
# 

# In[26]:


obesity_data["model_predictions"] = MODEL.predict()


# 
# Histograms, density plots, and scatter plots can then be made using these 
# variables as normal (see earlier code example for help).
# 

# 
# *COMPLETE THE FOLLOWING*
#   - Create the two plots described above (histogram of residuals, and scatter 
#     plot of residuals by model predictions).
#   - After creating these two plots , comment on whether there are any strong 
#     patterns, and if so, what they might indicate.
# 

# In[27]:


sns.histplot(x = "residuals", data=obesity_data)


# In[28]:


sns.scatterplot(x="model_predictions", y="residuals", data=obesity_data)


# There are no strong patterns

# 
# ## 2. Report null hypothesis significance tests for parameters
#   
# *COMPLETE THE FOLLOWING*
#   - Report if any of the slope (not Intercept) parameters appear to be 
#     statistically significant.
# 

# *COMPLETE THE FOLLOWING*
#   - Report if any of the slope (not Intercept) parameters appear to be 
#     statistically significant.
# 

# One way to examine tests for each parameter is to look at the coefficient table that is part of the summary created by the `summary()` method. In order to just display/print the coefficient table, and not all the other info in the `summary()`, you can do the following:
# 

# In[29]:


MODEL.summary().tables[1]


# The column `P>|t|` gives the p-value for the t-test of that parameter.
# 
# Alternatively, you can get just the p-values directly with:
# 

# In[30]:


MODEL.pvalues


# *COMPLETE THE FOLLOWING*
#   - Report the tests based on what you see in the results.
#   - Generally people don't report the Intercept p-value, since that's rarely of interest.
#   - So focus on reporting the statistical significance (based on the p-value) for each of the predictors.
#   - Specifically, if the p-value is less than 0.05, you can say:
#     - "The slope of [PRED] was [estimate], and was statistically significant 
#       from zero at the p < .05 level."
# 	- or: "The effect of [PRED] reached statistical significance at the p < .05 
#       level."
# 

# The p-values for the Q046 and Q036 are both very statistically significant at the p < .05 level. Q046 is .000 which means that it is certain that the percentage of aduts who do 300+ minutes of exercise per week has an effect on the percentage of adults with obesity. Q043 is .001 which means that it is almost certain that the percentage of aduts who do 2 muscle-strengthening exercises per week has an effect on the percentage of adults with obesity.
# 
# The slopes for both the Q046 and Q043 are both statistically significant. The coefficient for percent of adults doing 300+ minutes of exercise per week is -1.0053, which tells us that (essentially) for every percentage point more that people do 300+ minutes of exercise per week, the percent of adults with obesity goes down 1 percentage point. It almost a perfect 1 to -1 relationship. The coefficient for percent of adults who do 2+ days of muscle strengthening activities per week is -1.6863, which tells us that for every percentage point more that adults do 2 muscle strengthening activites per week, the percent of adults with obesity goes down by about 1.7 percent. 

# 
# **IF YOU ARE DOING MULTIPLE REGRESSION**:
#   - Report on each significant effect.
#   - Print the full table of coefficients (using the same `MODEL.summary().tables[1]` method as above)
# 

# In[31]:


MODEL.summary().tables[1]


# The percentage of adults who do 300+ minutes of exercise per week is a signifigant effect because it affects the obesity percentage by essentially 1 to -1 ratio. The percentage of adults who do 2+ days of muscle strengthening activites per week is a significant effect because it affects the obesity percentage by slightly less than a 1 to -1.7 ratio.

# 
# **IF YOU ARE DOING LOGISTIC REGRESSION**:
#   - The process is the same, it's just that the p-value is from a Wald's z-test, 
#     not a t-test.
# 
# 

# ## 3. Model comparison
# 
# *COMPLETE THE FOLLOWING*
#   - Compare your model to a model with no predictors, just the intercept. (You 
#     will need to fit this model before you can make comparisons.)
#   - Compare these two models with a test of statistical significance, and by 
#     comparing their AIC model fit statistics.
#   - To compare them with AIC, simply compute the AIC value of each model, and 
#     report those values. The lowest value indicates the better-fitting model.
#   
# To fit a model without any predictors, use a formula with just a "1" on the 
# predictor side, representing the intercept:

# In[32]:


MODEL_null = smf.ols("Q036 ~ 1", data = obesity_data).fit()
# LOGISTIC_MODEL_null <- smf.logit("RESPONSE ~ 1", data = YOURDATA).fit()


#   
# To compute the AIC value of a model called MODEL:
# 

# In[33]:


MODEL.aic


# In[34]:


MODEL_null.aic


# The original model is the better fitting model because the AIC is lower

# To perform a Wald test (aka F-test) using the `statsmodels` library, you can use a function that lets you specify hypotheses. In this case, in order to compare the larger model (which has one or more predictors in it) to a model with no predictors (only the intercept), then you need to specify an hypothesis in which every predictor's slope parameter is zero. In other words, the null hypothesis for all predictors.
# 
# For example, if you had two predictors `zPRED1` and `zPRED2` in a model called MODEL, you could get an F-test with:

# In[35]:


MODEL.f_test("(cQ046 = 0, cQ043 = 0)")


# 
# *COMPLETE THE FOLLOWING*
#   - To report the results of the F-test, you need:
#     - the degrees of freedom (DoF): there are two DoFs for an F-test, the
#       "numerator" DoF and the "denominator" DoF
#     - the F-value
#     - the p-value
#   - For example, if the numerator DoF is 2, the denominator DoF is 50, 
#     the F is 4.25, and p-value is < .05, you would say:
#     - "An F-test (or Wald test) comparing [larger model] to [smaller model] was
#       statistically significant at the p < .05 level (F[2, 50] = 4.25), indicating
#       that the [larger model] was a better model than the [smaller model]."
#   - To find these values in the output of the `f_test()` method:
#     - `df_num` is the "numerator degrees of freedom"
#     - `df_denom` is the "denominator degrees of freedom"
#     - `F` is the F-value
#     - `p` is the p-value
# 
# 

# An F-test comparing MODEL_null to MODEL was statistically significant at the p < .05 level (F[2, 155] = 90.011), indicating that the original model (larger) was a better model than the null model (smaller model).

# ## 4. Interpretation
# 
#   - This is the hardest part, because it requires some thinking.
#   - Think about the questions you started with. Why does this analysis matter? 
#     Why is it interesting that a predictor (significantly) predicts the outcome?
# 

# *COMPLETE THE FOLLOWING*
#   - In order to help address these questions, phrase the effect of predictors not 
#     in terms of statistical significance (you already did that above), but in terms 
#     of the size of the effect.
#   - To compute an effect size, first think about what the predictor represents, 
#     and think about what would be a meaningful level of change in that predictor.
#   - Unless you have transformed the variable, the slope estimate you get from your
#     model means "for every 1 unit change in X, the model predicts a change of
#     [estimate] in Y". But when reporting the results, you should think about whether
#     "1 unit of change in X" makes sense, or whether it would be best to scale that
#     up or down.
#   - For example, if you were discussing the effect of the weight of a diamond on
#     price, and your regression used weight in carats, then the parameter estimate
#     would mean "the effect on price from a change of 1 carat." But a change of 1
#     whole carat is huge, when you are talking about diamonds!  It might make more
#     sense to talk about a half-carat or quarter-carat.  In other words, you might
#     like to say something like, "the change in price by going up in weight by 0.25
#     carats is estimated to be [X] dollars."
#   - In order to calculate the "[X]" in "[X] dollars" in this example, you would
#     take the parameter estimate and divide it by 4. In other words, since the 
# 	  parameter estimate always means "change in Y from a 1-unit change in X", if
#     you want to get the change in Y from a change in 1/4 units of X, you need to
#     divide the estimate by 4 as well.  Similarly, if you wanted to calculate the 
#     change in Y from a change in 10 units of X, you'd multiply the model parameter 
#     estimate by 10.
#   - If you transformed your predictors into z-scores for the model-fitting, then
#     remember that "one unit of X" actually means "1 standard deviation of X". You
#     can report the effect this way, but then you should also remind the reader
#     what 1 standard deviation corresponds to.
#   - For example, in the hypothetical diamond analysis, if your model used z-scores
#     of weight instead of raw carat weight, and the standard deviation of carat was 
#     0.5 carats, you could say:
#     - "For a change in 1 standard deviation of weight (0.5 carats), the model
#       predicts a change of [X] dollars in price."
# 

# The slopes for both the Q046 and Q043 are both statistically significant. The coefficient for percent of adults doing 300+ minutes of exercise per week is -1.0053, which tells us that (essentially) for every percentage point more that people do 300+ minutes of exercise per week, the percent of adults with obesity goes down 1 percentage point. It almost a perfect 1 to -1 relationship. The coefficient for percent of adults who do 2 muscle strengthening activities per week is -1.6863, which tells us that for every percentage point more that adults do 2 muscle strengthening activites per week, the percent of adults with obesity goes down by about 1.7 percent. 

# **IF YOU ARE DOING MULTIPLE REGRESSION**
#   - Report the effect of each predictor, or combine them.
#   - If you have several predictors and only some of them are statistically
#     significant, you can report just the significant effects.
#   - If you want to combine them, you could say something like, "if you increased
#     [PRED1] by [X1], [PRED2] by [X2], and [PRED3] by [X3], the overall predicted
#     change in [Y] would be [result]."
# 

# If you increase Q046 (% of adults who do 300+ minutes of exercise per week) by 1, the predicted change to the Q036 (% of adults with obesity) would be -1.0053.
# 
# If you increase Q043 (% of adults who do 2+ days of muscle strengthening exercises per week) by 1, the predicted change to the Q036 (% of adults with obesity) would be -1.6863.

# **IF YOU ARE DOING LOGISTIC REGRESSION**
#   - This is the hardest part of logistic regression, because the parameters are
#     in logit space, which is not typically something that people easily translate
#     in their heads, so you need to transform it percentages/probabilities.
#   - In order to do this, there are two steps:
#     1. Transform the predictions from logit space to odds-ratio space, by using
#        `exp()`, the inverse of `log()`.
#     2. Transform the odds-ratio value to a percentage, which can be interpreted as
#        a probability.
#   - In other words, the parameters of a logistic model -- including Intercept and
#     slopes -- are essentially a log-transform of an odds ratio. So you need to
#     "undo" the log-transformation by using the `np.exp()` function. This is
#     described in more detail below under the "if you transformed your variables"
#     section.
#   - After doing that, the value will be an odds ratio. This is the kind of number
#     people use when they say something like "four-to-one odds" (or "4:1 odds") to
# 	  mean that one thing is twice as likely as another thing. The "ratio" of the
#     probabilities in this case is 4, so this corresponds to something having a
#     80% chance (since that's four times as likely as the 20% chance of something
#     else happening).
#   - To help you convert from an odds ratio to a percentage, I have defined the
#     following function (and you can see that it works for an odds ratio of 4):

# In[36]:


def odds_to_percentage(odds):
    percentage = odds/(1+odds)
    return percentage

odds_to_percentage(4)


# 
# So to recap, if you are doing logistic regression:
#   - Read the sections below.
#   - Compute the effects and plus/minus uncertainty as described below, using
#     the intercept, slope(s), and standard errors from your model.
#   - This might result in something like:
#     - "The effect of [PRED] on the likelihood of being in a Major key was such
#       that a change of 1 [unit] of [predictor variable] resulted in raising the
#       probability from [Intercept] to [Intercept plus effect], with a range of
#       uncertainty between [Intercept plus effect minus 2 standard errors] and
#       [Intercept plus effect plus 2 standard errors]."
#   - Then for all of those values above that include Intercepts and effects, you
#     need to convert them from logit values to percentages, by first applying
#     `np.exp()` to the logit value to convert it to an odds ratio, then applying
#     the `odds_to_percentage()` function to convert to percentages.
# 

# 
# *COMPLETE THE FOLLOWING*
#   - When reporting your effects, you also need to report something about the
#     uncertainty.
#   - This typically takes the form of giving a "plus/minus" or a range around the
#     estimate.
#   - For example, you might say something like, "for a change of 1 [unit] in X, the
#     model predicts we would see a change of [estimate] in Y, plus or minus [E]."
#   - The question is how to determine the width of the plus/minus. One standard
#     option is to add and subtract 2 standard errors. This is related to the
#     idea of a confidence interval, because +/- 2 standard errors is roughly the
#     same as a 95% confidence interval. So I recommend you use this when you are
#     reporting effects in "real world impact" terms.
#   - For example, if the effect of a change in 1 unit of X (i.e., the parameter
#     estimate from the model) is 10, with a standard error of 3, you would say
#     something like:
#     - "for a change in 1 [unit] of [X variable], the model predicts an average
#       change in 10 [units] of [Y variable], plus or minus 6 (i.e., a change between 
#       4 and 16 [units])."
# 	  
# 

# Q046 standard error: .297
# Q043 standard error: .297
# 
# For a change in 1 percentage point of Q046 (% of adults who do 300+ minutes of exercise per week), the model predicts an average change in -1.0053 percentage points of Q036 (% of adults with obesity), plus or minus .594 percentage points.
# 
# For a change in 1 percentage point of Q043 (% of adults who do 2+ days of muscle strengthening exercise per week), the model predicts an average change in -1.6863 percentage points of Q036 (% of adults with obesity), plus or minus .594 percentage points.

# **IF YOU TRANSFORMED YOUR VARIABLE(S)**
#   - If you transformed your variables, you will need to transform the results back
#     to something that is easier to interpret.
#   - For example, if you log-transformed a response variable that was originally in
#     dollars, you will need to transform the results back to dollars to make the
#     results more understandable.
#   - The function `exp()` is the inverse of `log()`, so if you need to take a log-
#     transformed variable and put it back on the original scale, use `np.exp()`.
#   - Additionally, you will not be able to interpret the results without adding in
#     the intercept, because if you compute changes in the log scale without *first*
#     adding the intercept, the calculations will be way off.
#   - As a full example, imagine you are using weight (in carats) to predict the
#     price of diamonds, but you log-transformed the price before fitting the model.
#     - Also assume we standardized (converted to z-scores) the weight predictor.
#     - Now let's say the model results indicated an Intercept of 8.5 and a parameter
#       estimate of 1.5 for the effect of weight with a standard error of 0.3.
#     - We can interpret this as meaning that for every change of 1 standard deviation 
#       in weight, the model predicts a change of 1.5 log-dollars. But this is hard to
#       understand, so it's our job to convert these numbers back to units that people
#       can more easily understand.
#     - The impact of going up a standard deviation from an average diamond would 
#       be a change of 1.5, but since we need to include the intercept when making our 
#       conversions, that 1.5 takes us from an 8.5 (the intercept) to a 10. Exp(8.5) 
#       is close to $5000, and exp(10) is around $22,000.
#     - Note that we can't just compute `np.exp(1.5)` (which is about 4.5) and say that
#       the effect of going up a standard deviation in weight results in raising the
#       price by $4.50. This is because we forgot to include the intercept. But if we
#       are thinking about the interpretation, this seems clearly wrong, in terms of 
#       how the prices of diamonds work!
#     - In short, if we are trying to convert results from log space back to regular 
#       units, we can't ignore the intercept in our calculations, and we have to add 
#       the effects to the intercept *before* we do our exponential conversion.
#     - If we use +/- 2 standard errors for our uncertainty, and our standard error 
#       is 0.3, then we add a plus/minus 0.6 to the prediction.
#     - In this example, it would mean that the predicted change would be from an 
#       8.5 (our intercept) to 10 (because we added the effect of 1.5), and then we
#       add +/- 0.6, meaning the final prediction is between 9.4 and 10.6. To get
#       these values from the log scale back to dollars, we'd end up with exp(9.4)
#       and exp(10.6), which would be around $12,000 and $40,000, respectively.
#     
#   - So the final reported statement would read something like the following:
# 	- "In our model, as weight increased, so did price, which was not surprising.
#       In our data, an average diamond weighed around 1 carat. For a diamond that
#       weighed 1 standard deviation (roughly 0.5 carats) more than average, the
#       model predicts an expected price of roughly $22,000, within a range of
#       uncertainty with 95% confidence between $12,000 and $40,000."
#   - Work through any conversions you need to, and try to report your results
#       in a similar manner.
# 

# **IF YOU ARE DOING MULTIPLE REGRESSION**
#   - Make sure you report this "plus/minus" uncertainty when reporting the effects
#     of each predictor you reported above.
# 

# In[37]:


# average obesity as a percentage
obesity_data["Q046"].mean()


# In[38]:


-1.0053 * (.29375949367088623 + .297) + 28.5285


# In[39]:


-1.6863 * (.29375949367088623 + .297) + 28.5285


# In our model, as Q046 (% of adults who do 300+ minutes of exercise per week) increased, Q036 (% of adults with obesity) decreased, which was not surprising. For every 1% more of adults who do 300+ minutes of exercise per week, the amount of adults with obesity will go down by an average of -1.0053% percent, plus or minus .594%.
# 
# In our model, as Q043 (% of adults who do 2+ days of muscle strengthening exercises per week) increased, Q036 (% of adults with obesity) decreased, which was not surprising. For every 1% more of adults who do 2+ days of muscle strengthening exercise per week, the amount of adults with obesity will go down by an average of -1.6863% percent, plus or minus .594%.
# 
# In our data, the average percent of american adults with obesity was 29.38%. For a percentage of adults who do 300+ minutes of exercise per week that is 1 standard deviation (roughly 0.297%) more than average, the model predicts the percentage of adults with obesity to be 27.93%.
# 
# In our data, the average percent of american adults with obesity was 29.38%. For a percentage of adults who do 2+ days of muscle strengthening exercises per week that is 1 standard deviation (roughly 0.297%) more than average, the model predicts the percentage of adults with obesity to be 27.53%.

# **IF YOU ARE DOING LOGISTIC REGRESSION**
#   - See the description above on converting the final quantities from logit
#     values to percentages, using `np.exp()` and `odds_to_percentage()`.
#   - Otherwise, compute the effects and range of uncertainty (plus/minus two
#     standard errors) the same way as described above for a linear model.
# 

# 
# *COMPLETE THE FOLLOWING*
#   - Now that you have provided estimated predictions along with a range of 
#     uncertainty, you need to comment on the importance of the finding. In short,
#     does this finding matter?
#   - There is no single right answer here, so just think about what the effects might
#     mean in the real world.
#   - This should circle back to the questions you had in Section 1.
#   - For example, if you were trying to predict the number of streams in Spotify,
#     are the effects large enough to care about?  Even if an effect was "statistically
#     significant", if the real-world meaning was that an artist would have to make
#     a big change in their music to get on average 10 more streams per year, then
#     that might not be worth it.  But if a change might results in millions of
#     additional streams, that might be.
#   - Again, there is no right or wrong answer here (for the purposes of this
#     assignment), but you should think about what the model results mean, think
#     about the effect sizes above, and reason about whether the results mean anything
#     that might be important to someone.
# 	
# 

# These findings are siginificant only because of the comparison between the predictors. Overall, the data just reinforces the current concensus in the medical community that exercise is a good way to avoid and combat obesity. However, this data shows that doing 2+ days of muscle strengthening activites per week has a stronger correlation to avoiding obesity than regular 300+ minutes of standard exercise per week. We can conclude from looking at this data that muscle strengthening exercises are more effective at combating obesity than standard exercise (such as aerobics). This conclusion could possibly be misleading; maybe most of the people who do muscle strengthening exercises also do standard aerobic exercise as well. Just looking at this data though, it is clear that muscle strengthening activites are more effective at combating obesity because the correlation coefficents for regular 300+ minutes of exercise and 2+ days of muscle strengthening exercises are -1.0053 and -1.6863 respectively; muscle strengthening exercises 2+ days a week is more than 50% stronger at lowering the percentage of adults with obesity.
# 
# People who want to either avoid obesity or exit their obese stage of life may find this information useful in determining what type of exercise they choose on their weight-loss journey. From the data, they would be more inclined to choose muscle strengthening exercises because it has a higher correlation to dropping obesity. Regular 300+ minutes of aerobic exercise is still effective as seen by the correlation coefficient of -1.0053, and should also be incorporated into the weight loss journey strategy.

# # Submission
# 
#   - Congratulations! You're done.
#   - (But if you're doing logistic regression, make sure to consult the additional
#     document.)
#   - Please submit your copy of this .Rmd file on ELMS as your final project.
#   - Thank you, and have a great winter break!
# 
