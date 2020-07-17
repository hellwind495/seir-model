import pandas as pd 
import numpy as np 
import scipy.stats as st 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns 
import datetime
#from scipy.interpolate import BSpline
import logging
from datetime import datetime
from datetime import timedelta
import logging
import argparse

from pathlib import Path

# survival analysis
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from matplotlib.lines import Line2D

# mortality analysis
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../Western Cape/',
                    help='Folder where the data is stored')
parser.add_argument('--data_date', type=str, default='2020/07/15',
                    help='Date on which data was extracted')
parser.add_argument('--age_groups', action='store_true')

def main():
    # parse args
    args = parser.parse_args()
    data_path = args.data_path
    data_date = pd.to_datetime(args.data_date)
    data_date_str = datetime.strftime(data_date,'%Y%m%d')
    line_date = Path(args.data_path + 'Covid-19 Anonymised line list ' + data_date_str + '.csv')
    
    # check args
    #assert data_path.exists() and data_path.is_file(), \
    #    f"Given data file '{args.data_file}' does not exist or is not a file."

    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -- %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    #####################
    # process line data #
    #####################

    # load data
    logging.info(f'Loading data from {args.data_path}')

    date_cols = ['Discharge_date', 'Date_of_ICU_admission', 'Admission_date','date_of_death'] # exclude data_of_diagnosis1 because it's not formatted as a date in the raw data
    df_WC = pd.read_csv(data_path + 'Covid-19 Anonymised line list ' + data_date_str + '.csv',
                        parse_dates=date_cols)
    df_WC.rename(columns={'date_of_diagnosis1':'date_of_diagnosis'},inplace=True)
    df_WC['date_of_diagnosis'] = df_WC['date_of_diagnosis'].apply(lambda x: pd.to_datetime(x).date())
    logging.info(f'Rows in line data: {df_WC.shape[0]:,}.')

    # remove non-hospitalised and invalid records
    df_WC = df_WC[~pd.isna(df_WC['Admission_date'])]
    logging.info(f'Rows after filtering for hospitalisations: {df_WC.shape[0]:,}.')
    df_WC = df_WC[~(df_WC['Discharge_date'] < df_WC['Admission_date'])]
    df_WC = df_WC[~(df_WC['Admission_date'] > data_date)].reset_index()
    logging.info(f'Rows after removing discharge > admission or admission > today: {df_WC.shape[0]:,}.')

    # insert admission_status and admitted_to_icu
    df_WC['admission_status'] = df_WC.apply(admission_status, axis=1)
    df_WC['admitted_to_icu'] = df_WC.apply(admitted_to_icu, axis=1)

    # remove leading ' from agregroup
    df_WC['agegroup'] = [x[1:] for x in df_WC['agegroup']]

    # construct date range
    min_date = df_WC[date_cols].min().min()
    max_date = data_date
    logging.info(f'Constructing data from {min_date} to {max_date}')
    date_range = pd.date_range(
        start=min_date,
        end=max_date
    )

    # prepare output df
    df_out = pd.DataFrame({'date': date_range, 'Current Hospitalisations': 0, 'Current ICU': 0,
                        'Cum Deaths': 0, 'Cum Recoveries': 0})

    logging.info('Calculating...')
    for date in date_range:
        df_hosp_current = df_WC.apply(current_hospital_patient, axis=1, date=date)
        df_hosp_current_excl_discharge = df_WC.apply(current_hospital_patient_excl_discharge_date, axis=1, date=date)
        df_icu_current = df_WC.apply(current_icu_patient, axis=1, date=date)
        df_icu_current_excl_discharge = df_WC.apply(current_icu_patient_excl_discharge_date, axis=1, date=date)
        df_deaths = df_WC.apply(current_deaths, axis=1, date=date)
        df_recoveries = df_WC.apply(current_recoveries, axis=1, date=date)

        df_out.loc[df_out['date'] == date, 'Current Hospitalisations'] = df_hosp_current.sum()
        df_out.loc[df_out['date'] == date, 'Current ICU'] = df_icu_current.sum()
        df_out.loc[df_out['date'] == date, 'Current Hospitalisations excl discharge date'] = df_hosp_current_excl_discharge.sum()
        df_out.loc[df_out['date'] == date, 'Current ICU excl discharge date'] = df_icu_current_excl_discharge.sum()        
        df_out.loc[df_out['date'] == date, 'Cum Deaths'] = df_deaths.sum()
        df_out.loc[df_out['date'] == date, 'Cum Recoveries'] = df_recoveries.sum()

    save_path = Path(args.data_path + 'WC_data_' + data_date_str + '.csv')
    logging.info(f"Saving data to '{save_path}'")
    df_out.to_csv(save_path, index=False)

    ##########################################
    # compare to aggregated calibration data #
    ##########################################

    calib = pd.read_csv(data_path + 'Covid-19 calibration data ' + data_date_str + '.csv'
                    ,parse_dates=['date'],date_parser=lambda x:pd.to_datetime(x,format='%d %m %Y'))

    # filter out before earliest common date
    min_date = max(df_out['date'].min(),calib['date'].min())
    df_out = df_out[df_out['date'] >= min_date]
    calib = calib[calib['date'] >= min_date]

    fig, ax = plt.subplots(1,2,figsize=(20,5))

    ax[0].plot(df_out['date'],df_out['Current Hospitalisations'],label='Derived from line data')
    ax[0].plot(df_out['date'],df_out['Current Hospitalisations excl discharge date'],label='Derived, discharge date excluded')
    ax[0].plot(calib['date'],calib['current_general'],label='Calibration data')
    ax[0].legend()
    ax[0].xaxis.set_major_locator(mdates.MonthLocator())

    ax[1].plot(df_out['date'],df_out['Current ICU'],label='Derived from line data')
    ax[1].plot(df_out['date'],df_out['Current ICU excl discharge date'],label='Derived, discharge date excluded')
    ax[1].plot(calib['date'],calib['current_ICU'],label='Calibration data')
    ax[1].legend()
    ax[1].xaxis.set_major_locator(mdates.MonthLocator())

    ax[0].set_title('Hospital general')
    ax[1].set_title('ICU')
    fig.suptitle('Comparison of calibration data with figures derived from line data')

    fig.savefig(data_path + 'line_calib_comparison_' + data_date_str + '.png')

    ##################################
    # Kaplan-Meier survival analysis #
    ##################################

    if args.age_groups:
        print('Doing KM estimate for multiple age groups')
        age_groups = [['0 - 5', '5 - 10'], ['10 - 15', '15 - 20'], ['20 - 25', '25 - 30'], ['30 - 35', '35 - 40'],
                    ['40 - 45', '45 - 50'], ['50 - 55', '55 - 60'], ['60 - 65', '65 - 70'], ['70 - 75', '75 - 80'],
                    ['80 - 85', '85 - 90', '90 - 95', '95 - 100']]
        suffixes = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
        for suffix, group in zip(suffixes, age_groups):
            print(f'Calculating KM for {suffix}')
            filter = df_WC['agegroup'] == group[0]
            for i in range(1, len(group)):
                filter = (filter) | (df_WC['agegroup'] == group[i])
            km_estimate(df_WC[filter].reset_index(), fig_path=f'{data_path}KM_estimates_{suffix}.png', csv_path=f'{data_path}durations_{suffix}.csv')
    else:
        km_estimate(df_WC, fig_path=f'{data_path}KM_estimates_allages.png', csv_path=f'{data_path}durations_allages.csv')


    ######################
    # mortality analysis #
    ######################

    df, df_h, df_c = data_adj_for_mort(df_WC)

    # get total deaths for adjustment
    df_deaths = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv',
        parse_dates=['date'],
        date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
    )
    death_total = df_deaths.loc[df_deaths['date']==max_date,'WC'].values[0]
    death_hospital = df[df['admission_status']=='Died'].shape[0]
    death_adjt = death_total / death_hospital

    mort,mort_h,mort_c = pivot_mort(df,df_h,df_c,data_path,death_adjt)

    plot_mortality(mort,data_path,data_date_str)
    plot_mort_hosp_icu(mort_h,mort_c,data_path,data_date_str)

    mort.to_csv(data_path + 'mort' + data_date_str + '.csv')
    mort_h.to_csv(data_path + 'mort_h' + data_date_str + '.csv')
    mort_c.to_csv(data_path + 'mort_c' + data_date_str + '.csv')


#############
# Functions #
#############

def pivot_mort(df,df_h,df_c,data_path,death_adjt):

    mort = pd.pivot_table(df,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
    mort['Cases'] = mort['Died'] + mort['Discharged']
    mort['Adj_deaths'] = mort['Died'] 
    mort['Mid_age'] = [5,15,25,35,45,55,65,76,85]

    mort_h = pd.pivot_table(df_h,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
    mort_h['Cases'] = mort_h['Died'] + mort_h['Discharged']
    mort_h['Adj_deaths'] = mort_h['Died'] 
    mort_h['Mid_age'] = [5,15,25,35,45,55,65,76,85]

    mort_c = pd.pivot_table(df_c,values='Count',index='10yr_ageband',columns=['admission_status'],aggfunc=np.sum).fillna(0).reset_index()
    mort_c['Cases'] = mort_c['Died'] + mort_c['Discharged']
    mort_c['Adj_deaths'] = mort_c['Died'] 
    mort_c['Mid_age'] = [5,15,25,35,45,55,65,76,85]

    mort_rates = mort.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases'], alpha=0.05), axis=1)
    mort['p_low'] = [x[0] for x in mort_rates]
    mort['p_mean'] = [x[1] for x in mort_rates]
    mort['p_high'] = [x[2] for x in mort_rates]
    mort['p_scaled'] = mort['p_mean'] * death_adjt

    mort_rates_h = mort_h.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases'],alpha=0.05), axis=1)
    mort_h['p_low'] = [x[0] for x in mort_rates_h]
    mort_h['p_mean'] = [x[1] for x in mort_rates_h]
    mort_h['p_high'] = [x[2] for x in mort_rates_h]
    mort_h['p_scaled'] = mort_h['p_mean'] * death_adjt

    mort_rates_c = mort_c.apply(lambda row: conf_int(row['Adj_deaths'], row['Cases'], alpha=0.05), axis=1)
    mort_c['p_low'] = [x[0] for x in mort_rates_c]
    mort_c['p_mean'] = [x[1] for x in mort_rates_c]
    mort_c['p_high'] = [x[2] for x in mort_rates_c]
    mort_c['p_scaled'] = mort_c['p_mean'] * death_adjt

    ferguson = pd.read_excel(data_path + 'ferguson.xlsx',sheet_name='Sheet1')

    mort = mort.merge(ferguson[['age_group','ifr_from_hospital']],right_on='age_group',left_on='10yr_ageband')
    mort.rename(columns={'ifr_from_hospital':'Ferguson_p'},inplace=True)

    params, pcov = curve_fit(gompertz, xdata=mort['Mid_age'], ydata=mort['p_scaled'], p0=[0.6,10,0.05])
    mort['Gompertz'] = gompertz(mort['Mid_age'],params[0],params[1],params[2])
    params, pcov = curve_fit(gompertz, xdata=mort_h['Mid_age'], ydata=mort_h['p_scaled'], p0=[0.6,20,0.05])
    mort_h['Gompertz'] = gompertz(mort_h['Mid_age'],params[0],params[1],params[2])
    params, pcov = curve_fit(gompertz, xdata=mort_c['Mid_age'], ydata=mort_c['p_scaled'], p0=[0.7,20,0.1])
    mort_c['Gompertz'] = gompertz(mort_c['Mid_age'],params[0],params[1],params[2])

    return(mort,mort_h,mort_c)

def plot_mortality(mort,data_path,data_date_str):

    fig,ax = plt.subplots(1,1)

    ax.plot(mort['p_mean'],c='darkblue',label='WC estimate')
    ax.set_xlabel('Ageband')
    ax.set_xticks(mort.index)
    ax.set_xticklabels(mort['10yr_ageband'])
    ax.plot(mort['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
    ax.plot(mort['p_high'],c='darkblue',linestyle='dashed')
    ax.plot(mort['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
    ax.plot(mort['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
    ax.plot(mort['Ferguson_p'],c='red',linestyle='dashdot',label='Ferguson')
    ax.fill_between(mort.index,mort['p_low'],mort['p_high'],alpha=0.5,color='darkgray')

    ax.set_title('Comparison of Western Cape Covid-19 mortality rate estimates by age with Ferguson et al. (2020)')
    ax.legend()

    fig.savefig(data_path + 'Mortality_rates_' + data_date_str + '.png')

def plot_mort_hosp_icu(mort_h,mort_c,data_path,data_date_str):

    fig,ax = plt.subplots(2,1)

    ax[0].plot(mort_h['p_mean'],c='darkblue',label='WC estimate')
    ax[0].set_xlabel('Ageband')
    ax[0].set_xticks(mort_h.index)
    ax[0].set_xticklabels(mort_h['10yr_ageband'])
    ax[0].plot(mort_h['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
    ax[0].plot(mort_h['p_high'],c='darkblue',linestyle='dashed')
    ax[0].fill_between(mort_h.index,mort_h['p_low'],mort_h['p_high'],alpha=0.5,color='darkgray')
    ax[0].plot(mort_h['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
    ax[0].plot(mort_h['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
    ax[0].set_title('Mortality rates out of hospital')
    ax[0].legend()

    ax[1].plot(mort_c['p_mean'],c='darkblue',label='WC estimate')
    ax[1].set_xlabel('Ageband')
    ax[1].set_xticks(mort_c.index)
    ax[1].set_xticklabels(mort_c['10yr_ageband'])
    ax[1].plot(mort_c['p_low'],c='darkblue',linestyle='dashed',label='WC 95% confidence bounds')
    ax[1].plot(mort_c['p_high'],c='darkblue',linestyle='dashed')
    ax[1].fill_between(mort_c.index,mort_c['p_low'],mort_c['p_high'],alpha=0.5,color='darkgray')
    ax[1].plot(mort_c['p_scaled'],c='darkgreen',linestyle='dashdot',label='Scaled up')
    ax[1].plot(mort_c['Gompertz'],c='violet',linestyle='dashed',label=f'Gompertz fitted to scaled')
    ax[1].set_title('Mortality rates out of ICU')
    ax[1].legend()

    fig.tight_layout()

    fig.savefig(data_path + 'Mortality_rates_hosp_icu_' + data_date_str + '.png')


def gompertz(x,a,b,c):
    return (a*np.exp(-b*np.exp(-c*x)))


def data_adj_for_mort(df):

    # drop last 5 days of information given reporting delays
    max_date = df['Admission_date'].max() - timedelta(days=5)
    df = df[df['Admission_date'] <= max_date]
    df.loc[df['Discharge_date'] > max_date,'admission_status'] = 'Inpatient'
    print(f'Number of records after removing last 5 days: {df.shape[0]:,}')
    print(df['admission_status'].value_counts())

    # drop no age or inpatients
    df = df[df['agegroup'] != 'Unknown']
    print(f'After dropping records with no age recorded: {df.shape[0]:,}')
    df = df[df['admission_status'] != 'Inpatient']
    print(f'After dropping current inpatients: {df.shape[0]:,}')
    print(df['admission_status'].value_counts())

    # change death status to died for those who died after discharge
    print('Before changing death status:\n')
    print(df['admission_status'].value_counts())
    df.loc[~pd.isna(df['date_of_death']),'admission_status'] = 'Died'
    print('\nAfter changing death status:\n')
    print(df['admission_status'].value_counts())

    df['Count'] = 1

    df['10yr_ageband'] = df['agegroup'].apply(tenyr_ageband)

    df_h = df[df['admitted_to_icu']=='No']
    df_c = df[df['admitted_to_icu']=='Yes']

    return(df,df_h,df_c)


def tenyr_ageband(agegroup):
    lower = int((int(agegroup[:2])+2)/10)*10
    if lower >= 80:
        return('80+')
    else:
        return(str(lower) + ' - ' + str(lower+9))


def conf_int(deaths,n,alpha):
    z = st.norm.ppf(1 - alpha/2)
    p = deaths/n
    if (n > 15) and (p > 0.1):
        # normal approximation
        sampling_err = np.sqrt(p*(1-p)/n)
        lo = max(0,p - z * sampling_err)
        hi = min(1,p + z * sampling_err)
    else: #elif (n > 15) and (p <= 0.1):
        # Poisson approximation
        if deaths == 0:
            lo = 0
        else:
            lo = st.chi2.ppf(alpha/2, 2*deaths) / (2 * n)
        hi = min(1,st.chi2.ppf(1 - alpha/2, 2*deaths + 2) / (2 * n))
    #else: binomial search, ignored for now
    return lo,p,hi


def filter_covid_hospital_cases(row):  # note 21-day post-discharge diagnosis rule for determining Covid-19 hospitalisations
    if pd.isna(row['Admission_date']) or (row['date_of_diagnosis1'] - row['Discharge_date']).days > 21:
        return False
    return True

def current_hospital_patient(row, date):
    hospital_case = False
    if row['Admission_date'] <= date and (row['Discharge_date'] >= date or pd.isna(row['Discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['admitted_to_icu'] == 'No' or date <= row['Date_of_ICU_admission']:
            hospital_case = True
    return hospital_case

def current_hospital_patient_excl_discharge_date(row, date):
    hospital_case = False
    if row['Admission_date'] <= date and (row['Discharge_date'] > date or pd.isna(row['Discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['admitted_to_icu'] == 'No' or date <= row['Date_of_ICU_admission']:
            hospital_case = True
    return hospital_case

def admission_status(row):
    if pd.isna(row['date_of_death']) and pd.isna(row['Discharge_date']):
        return 'Inpatient'
    elif pd.isna(row['date_of_death']) and pd.notna(row['Discharge_date']):
        return 'Discharged'
    elif pd.notna(row['date_of_death']):
        return 'Died'

def admitted_to_icu(row):
    icu = 'Yes'
    if pd.isna(row['Date_of_ICU_admission']):
        icu = 'No'
    return(icu)

def current_icu_patient(row, date):
    icu_case = False
    if row['Admission_date'] <= date and (row['Discharge_date'] >= date or pd.isna(row['Discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['admitted_to_icu'] == 'Yes' and date >= row['Date_of_ICU_admission']:
            icu_case = True
    return icu_case

def current_icu_patient_excl_discharge_date(row, date):
    icu_case = False
    if row['Admission_date'] <= date and (row['Discharge_date'] > date or pd.isna(row['Discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['admitted_to_icu'] == 'Yes' and date >= row['Date_of_ICU_admission']:
            icu_case = True
    return icu_case

def current_deaths(row, date):
    death = False
    if row['Discharge_date'] < date and row['admission_status'] == 'Died':
        death = True
    return death

def current_recoveries(row, date):
    recovery = False
    if row['Discharge_date'] < date and row['admission_status'] == 'Discharged':
        recovery = True
    return recovery


def durn(start,end,max):
    '''
    Calculate duration in days from start to end (if defined) or max (if end not defined)
    '''
    if pd.isna(start):
      return(-1)
    else:
      if pd.isna(end):
        end = max
      return (divmod((end - start).days, 1)[0])


def km_estimate(df,fig_path,csv_path):
    # calculate observed mortality rates for weighting right-censored data
    death_rate_hosp = df[(df['admission_status'] == 'Died') & (df['admitted_to_icu'] == 'No')].shape[0] / \
                      df[(df['admission_status'] != 'Inpatient') & (df['admitted_to_icu'] == 'No')].shape[0]
    death_rate_icu = df[(df['admission_status'] == 'Died') & (df['admitted_to_icu'] == 'Yes')].shape[0] / \
                     df[(df['admission_status'] != 'Inpatient') & (df['admitted_to_icu'] == 'Yes')].shape[0]

    # get start and end dates of data
    first_date = df.loc[~pd.isna(df['Admission_date']),'Admission_date'].min()
    last_date = df.loc[~pd.isna(df['Discharge_date']),'Discharge_date'].max()
    timespan = divmod((last_date - first_date).days, 1)[0] + 1

    # initialise duration, observed (Boolean) and weight lists
    n = df.shape[0]
    durn_hosp_to_discharge = [0] * n
    observed_hosp_to_discharge = [False] * n
    weight_hosp_to_discharge = [1] * n
    durn_hosp_to_death = [0] * n
    observed_hosp_to_death = [False] * n
    weight_hosp_to_death = [1] * n
    durn_hosp_to_icu = [0] * n
    observed_hosp_to_icu = [False] * n
    weight_hosp_to_icu = [1] * n
    durn_icu_to_discharge = [0] * n
    observed_icu_to_discharge = [False] * n
    weight_icu_to_discharge = [1] * n
    durn_icu_to_death = [0] * n
    observed_icu_to_death = [False] * n
    weight_icu_to_death = [1] * n

    # data checks
    death_wo_discharge_date = sum((df['admission_status'] == 'Died') & (pd.isna(df['Discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status'] == 'Discharged') & (pd.isna(df['Discharge_date'])))
    discharge_wo_discharge_date = sum((df['admission_status'] == 'Inpatient') & (~pd.isna(df['Discharge_date'])))
    icu_wo_icu_admission = sum((df['admitted_to_icu'] == 'Yes') & (pd.isna(df['Date_of_ICU_admission'])))
    print(f'Deaths without discharge dates: {death_wo_discharge_date}')
    print(f'Discharges without discharge dates: {discharge_wo_discharge_date}')
    print(f'Inpatients with discharge dates: {discharge_wo_discharge_date}')
    print(f'ICU without ICU admission dates: {icu_wo_icu_admission}')

    # assign survival times, observation flags and weight
    # TODO: make this more Pythonic!
    for i in range(n):
        if df.at[i, 'admission_status'] == 'Died':
            durn_hosp_to_discharge[i] = -1
            durn_icu_to_discharge[i] = -1
            if df.at[i, 'admitted_to_icu'] == 'No':
                durn_hosp_to_death[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Discharge_date'], last_date)
                durn_hosp_to_icu[i] = -1
                observed_hosp_to_death[i] = True
                durn_icu_to_death[i] = -1
            else:
                durn_hosp_to_death[i] = -1
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_death[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'Discharge_date'], last_date)
                observed_icu_to_death[i] = True
        elif df.at[i, 'admission_status'] == 'Discharged':
            durn_hosp_to_death[i] = -1
            durn_icu_to_death[i] = -1
            if df.at[i, 'admitted_to_icu'] == 'Yes':
                durn_hosp_to_discharge[i] = -1
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'Discharge_date'],
                                                last_date)
                observed_icu_to_discharge[i] = True
            else:
                durn_hosp_to_icu[i] = -1
                durn_icu_to_discharge[i] = -1
                durn_hosp_to_discharge[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Discharge_date'], last_date)
                observed_hosp_to_discharge[i] = True
        else:  # inpatients
            if df.at[i, 'admitted_to_icu'] == 'No':
                durn_hosp_to_icu[i] = -1  # assume none, given that most ICU cases are admitted directly
                durn_hosp_to_discharge[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Discharge_date'], last_date)
                durn_hosp_to_death[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Discharge_date'], last_date)
                weight_hosp_to_discharge[i] = 1 - death_rate_hosp
                weight_hosp_to_death[i] = death_rate_hosp
            else:
                durn_hosp_to_icu[i] = durn(df.at[i, 'Admission_date'], df.at[i, 'Date_of_ICU_admission'], last_date)
                observed_hosp_to_icu[i] = True
                durn_icu_to_discharge[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'Discharge_date'],
                                                last_date)
                durn_icu_to_death[i] = durn(df.at[i, 'Date_of_ICU_admission'], df.at[i, 'Discharge_date'], last_date)
                weight_icu_to_discharge[i] = 1 - death_rate_icu
                weight_icu_to_death[i] = death_rate_icu

    # populate dataframe from lists
    df['durn_hosp_to_discharge'] = durn_hosp_to_discharge
    df['observed_hosp_to_discharge'] = observed_hosp_to_discharge
    df['weight_hosp_to_discharge'] = weight_hosp_to_discharge
    df['durn_hosp_to_death'] = durn_hosp_to_death
    df['observed_hosp_to_death'] = observed_hosp_to_death
    df['weight_hosp_to_death'] = weight_hosp_to_death
    df['durn_hosp_to_icu'] = durn_hosp_to_icu
    df['observed_hosp_to_icu'] = observed_hosp_to_icu
    df['weight_hosp_to_icu'] = weight_hosp_to_icu
    df['durn_icu_to_discharge'] = durn_icu_to_discharge
    df['observed_icu_to_discharge'] = observed_icu_to_discharge
    df['weight_icu_to_discharge'] = weight_icu_to_discharge
    df['durn_icu_to_death'] = durn_icu_to_death
    df['observed_icu_to_death'] = observed_icu_to_death
    df['weight_icu_to_death'] = weight_icu_to_death

    # write duration data to CSV
    df.to_csv(csv_path)

    # Kaplan Meier estimates
    kmf_hosp_to_discharge = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_hosp_to_discharge'] != -1, 'durn_hosp_to_discharge'].values,
        event_observed=df.loc[df['durn_hosp_to_discharge'] != -1, 'observed_hosp_to_discharge'].values,
        weights=df.loc[df['durn_hosp_to_discharge'] != -1, 'weight_hosp_to_discharge'].values)
    kmf_hosp_to_death = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_hosp_to_death'] != -1, 'durn_hosp_to_death'].values,
        event_observed=df.loc[df['durn_hosp_to_death'] != -1, 'observed_hosp_to_death'].values,
        weights=df.loc[df['durn_hosp_to_death'] != -1, 'weight_hosp_to_death'].values)
    kmf_hosp_to_icu = KaplanMeierFitter().fit(durations=df.loc[df['durn_hosp_to_icu'] != -1, 'durn_hosp_to_icu'].values,
                                              event_observed=df.loc[
                                                  df['durn_hosp_to_icu'] != -1, 'observed_hosp_to_icu'].values,
                                              weights=df.loc[df['durn_hosp_to_icu'] != -1, 'weight_hosp_to_icu'].values)
    kmf_icu_to_discharge = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_icu_to_discharge'] != -1, 'durn_icu_to_discharge'].values,
        event_observed=df.loc[df['durn_icu_to_discharge'] != -1, 'observed_icu_to_discharge'].values,
        weights=df.loc[df['durn_icu_to_discharge'] != -1, 'weight_icu_to_discharge'].values)
    kmf_icu_to_death = KaplanMeierFitter().fit(
        durations=df.loc[df['durn_icu_to_death'] != -1, 'durn_icu_to_death'].values,
        event_observed=df.loc[df['durn_icu_to_death'] != -1, 'observed_icu_to_death'].values,
        weights=df.loc[df['durn_icu_to_death'] != -1, 'weight_icu_to_death'].values)

    n_hosp_to_discharge = sum((df['durn_hosp_to_discharge'] != -1) & (df['durn_hosp_to_icu'] == -1))
    n_hosp_to_discharge_censored = sum(df['weight_hosp_to_discharge'].between(0.01, 0.99))
    n_hosp_to_death = sum((df['durn_hosp_to_death'] != -1) & (df['durn_hosp_to_icu'] == -1))
    n_hosp_to_death_censored = sum(df['weight_hosp_to_discharge'].between(0.01, 0.99))
    n_hosp_to_icu = sum(df['durn_hosp_to_icu'] != -1)
    n_hosp_to_icu_censored = sum(df['weight_hosp_to_icu'].between(0.01, 0.99))
    n_icu_to_discharge = sum((df['durn_icu_to_discharge'] != -1) & (df['durn_hosp_to_icu'] != -1))
    n_icu_to_discharge_censored = sum(df['weight_icu_to_discharge'].between(0.01, 0.99))
    n_icu_to_death = sum((df['durn_icu_to_death'] != -1) & (df['durn_hosp_to_icu'] != -1))
    n_icu_to_death_censored = sum(df['weight_icu_to_death'].between(0.01, 0.99))

    mean_hosp_to_discharge = restricted_mean_survival_time(kmf_hosp_to_discharge, 100).mean()
    mean_hosp_to_death = restricted_mean_survival_time(kmf_hosp_to_death, 100).mean()
    mean_hosp_to_icu = restricted_mean_survival_time(kmf_hosp_to_icu, 100).mean()
    mean_icu_to_discharge = restricted_mean_survival_time(kmf_icu_to_discharge, 100).mean()
    mean_icu_to_death = restricted_mean_survival_time(kmf_icu_to_death, 100).mean()
    
    crit = np.exp(-1)

    hosp_to_discharge_crit = kmf_hosp_to_discharge.survival_function_[kmf_hosp_to_discharge.survival_function_['KM_estimate'] < crit].index.min()
    hosp_to_death_crit = kmf_hosp_to_death.survival_function_[kmf_hosp_to_death.survival_function_['KM_estimate'] < crit].index.min()
    hosp_to_icu_crit = kmf_hosp_to_icu.survival_function_[kmf_hosp_to_icu.survival_function_['KM_estimate'] < crit].index.min()
    icu_to_discharge_crit = kmf_icu_to_discharge.survival_function_[kmf_icu_to_discharge.survival_function_['KM_estimate'] < crit].index.min()
    icu_to_death_crit = kmf_icu_to_death.survival_function_[kmf_icu_to_death.survival_function_['KM_estimate'] < crit].index.min()

    # plot
    fig, axes = plt.subplots(5, 1, figsize=(20, 20))

    kmf_hosp_to_discharge.plot(ax=axes[0])
    kmf_hosp_to_death.plot(ax=axes[1])
    kmf_hosp_to_icu.plot(ax=axes[2])
    kmf_icu_to_discharge.plot(ax=axes[3])
    kmf_icu_to_death.plot(ax=axes[4])

    axes[0].set_title(
        f'Hospital to discharge for non-ICU cases: n = {n_hosp_to_discharge} including {n_hosp_to_discharge_censored} right-censored at weight {1 - death_rate_hosp:.3f}')
    axes[1].set_title(
        f'Hospital to death for non-ICU cases: n = {n_hosp_to_death} including {n_hosp_to_death_censored} right-censored at weight {death_rate_hosp:.3f}')
    axes[2].set_title(f'Hospital to ICU: n = {n_hosp_to_icu} including {n_hosp_to_icu_censored} right-censored ')
    axes[3].set_title(
        f'ICU to discharge: n = {n_icu_to_discharge} including {n_icu_to_discharge_censored} right-censored at weight {1 - death_rate_icu:.3f}')
    axes[4].set_title(
        f'ICU to death: n = {n_icu_to_death} including {n_icu_to_death_censored} right-censored at weight {death_rate_icu:.3f}')

    axes[0].axvline(kmf_hosp_to_discharge.median_survival_time_, linestyle='--', color='red')
    #axes[0].text(kmf_hosp_to_discharge.median_survival_time_,0.1,kmf_hosp_to_discharge.median_survival_time_)
    axes[1].axvline(kmf_hosp_to_death.median_survival_time_, linestyle='--', color='red')
    axes[2].axvline(kmf_hosp_to_icu.median_survival_time_, linestyle='--', color='red')
    axes[3].axvline(kmf_icu_to_discharge.median_survival_time_, linestyle='--', color='red')
    axes[4].axvline(kmf_icu_to_death.median_survival_time_, linestyle='--', color='red')

    axes[0].axvline(mean_hosp_to_discharge, linestyle='--', color='green')
    axes[1].axvline(mean_hosp_to_death, linestyle='--', color='green')
    axes[2].axvline(mean_hosp_to_icu, linestyle='--', color='green')
    axes[3].axvline(mean_icu_to_discharge, linestyle='--', color='green')
    axes[4].axvline(mean_icu_to_death, linestyle='--', color='green')

    axes[0].axvline(hosp_to_discharge_crit,linestyle='--',color='blue')
    axes[1].axvline(hosp_to_death_crit, linestyle='--', color='blue')
    axes[2].axvline(hosp_to_icu_crit, linestyle='--', color='blue')
    axes[3].axvline(icu_to_discharge_crit, linestyle='--', color='blue')
    axes[4].axvline(icu_to_death_crit, linestyle='--', color='blue')   

    v_lines = [Line2D([0], [0], color='blue', lw=1),
               Line2D([0], [0], color='red', linestyle='--', lw=1),
               Line2D([0], [0], color='green', lw=1)]

    for i in range(5):
        axes[i].set_xlim(-1, 30)
        axes[i].set_xticks(list(range(31)))
        axes[i].set_xticklabels(list(range(31)))
        axes[i].legend(v_lines, ['KM estimate', 'Median', 'Mean'])

    plt.tight_layout()
    fig.savefig(fig_path)


if __name__ == '__main__':
    main()
