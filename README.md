# Credit-Card-Approval
Banks receive a lot of applications for issuance of credit cards. Many of them rejected for many reasons, like high-loan balances, low-income levels, or too many inquiries on an individual’s credit report. Manually analyzing these applications is error-prone and a time-consuming process. This task can be automated with the power of machine learning. The project provides the credit card approval predictor using different machine learning algorithm , just like the real banks do.

![image](https://github.com/user-attachments/assets/88f1a535-710c-432d-929c-48e106f28cad)

Consider the data present in the Credit Card Approval dataset file.
Following is the attribute related information:

ID: Unique Id of the row
CODE_GENDER: Gender of the applicant. M is male and F is female.
FLAG_OWN_CAR: Is an applicant with a car. Y is Yes and N is NO.
FLAG_OWN_REALTY: Is an applicant with realty. Y is Yes and N is No.
CNT_CHILDREN: Count of children.
AMT_INCOME_TOTAL: the amount of the income.
NAME_INCOME_TYPE: The type of income (5 types in total).
NAME_EDUCATION_TYPE: The type of education (5 types in total).
NAME_FAMILY_STATUS: The type of family status (6 types in total).
DAYS_BIRTH: The number of the days from birth (Negative values).
DAYS_EMPLOYED: The number of the days from employed (Negative values). This column has error values.
FLAG_MOBIL: Is an applicant with a mobile. 1 is True and 0 is False.
FLAG_WORK_PHONE: Is an applicant with a work phone. 1 is True and 0 is False.
FLAG_PHONE: Is an applicant with a phone. 1 is True and 0 is False.
FLAG_EMAIL: Is an applicant with a email. 1 is True and 0 is False.
OCCUPATION_TYPE: The type of occupation (19 types in total). This column has missing values.
CNT_FAM_MEMBERS: The count of family members.
This is a csv file with credit record for a part of ID in application record. We can treat it a file to generate labels for modeling. For the applicants who have a record more than 59 past due, they should be rejected.

After reading the data, we have the following columns.

ID: Unique Id of the row in application record. MONTHS_BALANCE: The number of months from record time. STATUS: Credit status for this month. X: No loan for the month C: paid off that month 0: 1-29 days past due 1: 30-59 days past due 2: 60-89 days overdue 3: 90-119 days overdue 4: 120-149 days overdue 5: Overdue or bad debts, write-offs for more than 150 days.

Problem Statement
To create a model that can predict whether an individual’s application for a credit card will be accepted or not.

Steps Followed for the Project
To get a basic introduction of our project & What’s the business problem associated with it ?
We’ll start by loading and viewing the dataset.
To manipulate data, if there are any missing entries in the dataset.
To perform exploratory data analysis (EDA) on our dataset.
To pre-process data before applying machine learning model to the dataset.
To apply machine learning models that can predict if an individual’s application for a credit card will be accepted or not.




