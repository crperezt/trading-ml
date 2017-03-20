# -*- coding:utf-8 -*-
# This script will generate insider datasets from Form 4 data

import requests
from os import listdir
from os.path import isdir, join
import errno
from bs4 import BeautifulSoup
from config import FORMS_DATA_PATH
import pandas as pd
import numpy as np
import sys
import urllib2
import time


class SecCrawler():

    def __init__(self):
        self.hello = "Welcome to Sec Crawler!"
        self.company_list = pd.read_csv('company_list.csv')
        self.sector_dict = {'Consumer Discretionary': 'xly', 'Consumer Staples': 'xlp', \
                            'Energy': 'xle', 'Financials': 'xlf', 'Health Care': 'xlv', \
                            'Industrials': 'xli', 'Information Technology': 'xlk',      \
                            'Materials': 'xlb', 'Real Estate': 'xlre',                  \
                            'Telecommunication Services': 'xlt', 'Utilities': 'xlu'}
        self.ret_frame = pd.DataFrame()
        self.spy_ret_frame = pd.DataFrame()
        self.last_date = ''
    #def parse_form4(self, company_code, start_date, end_date):
    def parse_form4(self, start_date, end_date, returns=True):
        #start_date, end_date in YYYY-MM format
        #Iterate through files for company
        #Calculate monthly NBC, NBV from start_date to end_date
        #Generate list of 12-tuples with month-year
        #Fetch return for months following month-year of each 12-tuple
        #Generate datapoints for company:
         #COMPANY, 12-month NBC 12-tuple, 12-month NBV 12-tuple, Sector, MktCap, Return for month 13
        
        #get list of all companies in SEC data directory
        #company_list = [f for f in listdir(DEFAULT_DATA_PATH) if isdir(join(DEFAULT_DATA_PATH,f))]
        company_list = pd.read_csv('company_list.csv')
        self.spy_ret_frame = pd.read_json("https://api.tiingo.com/tiingo/daily/spy/prices?startDate=" + start_date + "&endDate=" + end_date + "&token=8a387055f2f4081b89abfc6b3044284e958f178e")
        self.last_date = end_date
        #self.last_date = str(int(end_date[0:4]) - 1) + end_date[4:]

        #generate list of months to include in data
        start_month = int(start_date[5:7])
        start_year = int(start_date[0:4])
        end_month = int(end_date[5:7])
        end_year = int(end_date[0:4])
        date_set = [] 
        while start_year <= end_year:
            if start_year == end_year:
                if start_month > end_month:
                    break
            if start_month < 10:
                date_set.append(str(start_year) + '-0' + str(start_month))
            else:
                date_set.append(str(start_year) + '-' + str(start_month))
            start_year = start_year + 1 if start_month == 12 else start_year
            start_month = start_month + 1 if start_month != 12 else 1
        date_set.sort()

        dataset_file = open('dataset_6_12_spy_alt_norm.csv', 'a+')
        dataset_file.write('COMPANY,MONTH,NBC1,NBC2,NBC3,NBC4,NBC5,NBC6,NBC7,NBC8,NBC9,NBC10,NBC11,NBC12,NBV1,NBV2,NBV3,NBV4,NBV5,NBV6,NBV7,NBV8,NBV9,NBV10,NBV11,NBV12,SECTOR,MKTCAP,RET1,RET6,RET12\n')
        
        #for each company c directory
        #traverse through all downloaded files
        for i, c in enumerate(company_list.SYMBOL):
            
            print "Processing " + c + ". Company " + str(i) + " of " + str(len(company_list.SYMBOL))

            if self.pull_returns(c) == False:
                "Failed to fetch return data for " + c + ", skipping..."
                continue

            cpath = join(FORMS_DATA_PATH, c, c, 'form4')
            csector = company_list.SECTOR[i]
            cmktcap = company_list.MKTCAP[i]
            file_list = [f for f in listdir(cpath)]
            if not file_list:
                continue
            
            #create dictionary for data
            #keyed by month, each entry will have the totals of transactions
            #for that month
            company_data = {}
            for t in date_set:
                company_data[t] = {'purch': 0, 'sales': 0, 'sb': 0, 'ss': 0}
            
            #for each file within a company directory
            #find all transactions and aggregate them by month
            #input the totals into company_data dictionary
            for j, f in enumerate(file_list):
                form = open(join(cpath,f))
                #print join(cpath,f)

                soup = BeautifulSoup(form.read(), 'lxml-xml')
                #find all non-derivative transactions in file
                for t in soup.find_all('nonDerivativeTransaction'):
                    try:
                        t_date = t.transactionDate.value.string[0:7]
                    except:
                        continue
                    if t_date not in date_set:
                        #print "Date out of range!"
                        continue
                    #some transactions are in fractions, so need to convert to
                    #float then int
                    if t.transactionShares:
                        num_shares = int(float(t.transactionShares.value.string))
                        if t.transactionAcquiredDisposedCode.value.string == 'A':
                            company_data[t_date]['sb'] = company_data[t_date]['sb'] + num_shares
                            company_data[t_date]['purch'] = company_data[t_date]['purch'] + 1
                        elif t.transactionAcquiredDisposedCode.value.string == 'D':
                            company_data[t_date]['ss'] = company_data[t_date]['sb'] + num_shares
                            company_data[t_date]['sales'] = company_data[t_date]['sales'] + 1
       
                ##### This block includes transactions of derivatives in dataset
                # for t in soup.find_all('derivativeTransaction'):
                #     t_date = t.transactionDate.value.string[0:7]
                #     if t_date not in date_set:
                #         #print "Date out of range!"
                #         continue
                #     if t.transactionShares:
                #         num_shares = int(float(t.transactionShares.value.string))
                #         if t.transactionAcquiredDisposedCode.value.string == 'A':
                #             company_data[t_date]['sb'] = company_data[t_date]['sb'] + num_shares
                #             company_data[t_date]['purch'] = company_data[t_date]['purch'] + 1
                #         elif t.transactionAcquiredDisposedCode.value.string == 'D':
                #             company_data[t_date]['ss'] = company_data[t_date]['sb'] + num_shares
                #             company_data[t_date]['sales'] = company_data[t_date]['sales'] + 1



            #sort the dictionary by date for dataset
            #to add return data for 12-month segments
            sorted_cd = company_data.items()
            sorted_cd.sort()
            #print "Sorted company data for " + c + "\n" + str(sorted_cd) + "\n"

            
            #Divide each quantity by the maximum quantity for the company throughout the dataset period
            #e.g., total stocks bought for a month divided by the maximum amount of stock bought in any month for that company
            max_purch = max([x[1]['purch'] for x in sorted_cd])
            max_sales = max([x[1]['sales'] for x in sorted_cd])
            max_sb = max([x[1]['sb'] for x in sorted_cd])
            max_ss = max([x[1]['ss'] for x in sorted_cd])

            for j, cd in enumerate(sorted_cd):
                try:
                    sorted_cd[j][1]['purch'] = sorted_cd[j][1]['purch']/float(max_purch)
                except:
                    pass
                try:
                    sorted_cd[j][1]['sales'] = sorted_cd[j][1]['sales']/float(max_sales)
                except:
                    pass
                try:
                    sorted_cd[j][1]['sb'] = sorted_cd[j][1]['sb']/float(max_sb)
                except:
                    pass
                try:
                    sorted_cd[j][1]['ss'] = sorted_cd[j][1]['ss']/float(max_ss)
                except:
                    pass


            #create dataset for company c
            #for each month, form tuple with data of next 11 months
            #and form a datapoint
            for j, cd in enumerate(sorted_cd):
                #if there are no 12 consecutive months after this month
                #finish
                if j + 13 > len(sorted_cd):
                    break

                #write company ticker and month of first month in seq (j)
                dataset_file.write(c + ',' + sorted_cd[j][0] + ',')
                nbc = []
                nbv = []
                for k in range (j,j+12):
                    #print "Computing stats for " + c + " for month " + str(sorted_cd[k][0])
                    
                    nbc_k = float(sorted_cd[k][1]['purch'] - sorted_cd[k][1]['sales'])
                    nbc.append(nbc_k)
                    nbv_k = float(sorted_cd[k][1]['sb'] - sorted_cd[k][1]['ss'])
                    nbv.append(nbv_k)

                    # try:
                    #     nbc_k = float(sorted_cd[k][1]['purch'] - sorted_cd[k][1]['sales'])/float(sorted_cd[k][1]['purch'] + sorted_cd[k][1]['sales'])                      
                    # except ZeroDivisionError:
                    #     nbc.append(0.0)
                    # else:
                    #     nbc.append(nbc_k)
                    # try:
                    #     nbv_k = float(sorted_cd[k][1]['sb'] - sorted_cd[k][1]['ss'])/float(sorted_cd[k][1]['sb'] + sorted_cd[k][1]['ss']) 
                    # except ZeroDivisionError:
                    #     nbv.append(0.0)
                    # else:
                    #     nbv.append(nbv_k)
                
                for n in nbc:
                    dataset_file.write(str(n) + ',')
                for n in nbv:
                    dataset_file.write(str(n) + ',')

                #fetch return for month 13
                if returns:
                    ret = self.get_return_13(c, sorted_cd[j+12][0], csector)
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + ',' + str(ret[0]) + ',' + str(ret[1]) + ',' + str(ret[2]) + '\n')
                else:
                    dataset_file.write(str(csector) + ',' + str(cmktcap) + '\n')

        dataset_file.close()
            


    def pull_returns(self, c):
        for attempts in range (20):
            try:
                self.ret_frame = pd.read_json("https://api.tiingo.com/tiingo/daily/" + c.lower() + "/prices?startDate=2011-01-01&endDate=2016-12-31&token=8a387055f2f4081b89abfc6b3044284e958f178e")
            except urllib2.HTTPError as err:
                if err.code == 429:
                    print "429 HTTP Error, sleeping for 21 minutes..."
                    time.sleep(1260)
                    print "Back online"
                elif err.code == 404:
                    print "Error 404, data unavailable, skipping...\n"
                    return False
                else:
                    typ, value, traceback = sys.exc_info()
                    print "Could not retrieve stock data\n" 
                    print str(typ)
                    print str(value)
                    print "\n Retrying. Attempt " + str(attempts)    
            else:
                return True
            finally:
                if attempts == 19:
                    print "Failed to fetch stock data\n"
                    return False
        


    def get_return_13(self, company, month, csector):

        month_int = int(month[5:7])
        month_first = month + '-01'

        last_days = {1: '31', 2: '28', 3: '31', 4: '30', 5: '31', 6: '30',
        7:'31', 8:'31', 9:'30', 10:'31', 11:'30', 12:'31'}

        month_last = month + '-' + last_days[month_int]

        sector_ticker = self.sector_dict[csector]
        #print "Fetching return data for sector index: " + sector_ticker + '\n'

        month6_int = month_int + 5
        if month6_int > 12:
            month6_int = month6_int - 12
            year6_int = int(month[0:4]) + 1
        else:
            year6_int = int(month[0:4])

        month12_int = month_int + 11
        year12_int = int(month[0:4])
        if month12_int > 12:
            month12_int = month12_int - 12
            year12_int = int(month[0:4]) + 1
        else:
            year12_int = int(month[0:4])
        month6_str = str(month6_int) if month6_int > 9 else '0' + str(month6_int)
        month12_str = str(month12_int) if month12_int > 9 else '0' + str(month12_int)

        month_last_6 = str(year6_int) + '-' + month6_str + '-' + last_days[month6_int]
        month_last_12 = str(year12_int) + '-' + month12_str + '-' + last_days[month12_int]

        # print "Month of return: " + month + '\n'
        # print "First of month: " + month_first + '\n'
        # print "Last of month: " + month_last + '\n'
        # print "Last of 6 months: " + month_last_6 + '\n'
        # print "Last of 12 months: " + month_last_12 + '\n'


        ret_list = []
        spy_ret_list = []
        ret_list.append(self.ret_frame[(self.ret_frame.date >= month_first) & (self.ret_frame.date <= month_last)])
        spy_ret_list.append(self.spy_ret_frame[(self.spy_ret_frame.date >= month_first) & (self.spy_ret_frame.date <= month_last)])
        ret_list.append(self.ret_frame[(self.ret_frame.date >= month_first) & (self.ret_frame.date <= month_last_6)])
        spy_ret_list.append(self.spy_ret_frame[(self.spy_ret_frame.date >= month_first) & (self.spy_ret_frame.date <= month_last_6)])
        ret_list.append(self.ret_frame[(self.ret_frame.date >= month_first) & (self.ret_frame.date <= month_last_12)])
        spy_ret_list.append(self.spy_ret_frame[(self.spy_ret_frame.date >= month_first) & (self.spy_ret_frame.date <= month_last_12)])

        ret = []

        try:
            price_rel_1 = float(ret_list[0].adjClose[ret_list[0].first_valid_index()])/float(spy_ret_list[0].adjClose[spy_ret_list[0].first_valid_index()])
            price_rel_30 = float(ret_list[0].adjClose[ret_list[0].last_valid_index()])/float(spy_ret_list[0].adjClose[spy_ret_list[0].last_valid_index()])
            ret.append((float(price_rel_30) - float(price_rel_1))/float(price_rel_1))

            #if np.str(ret_list[1].date[ret_list[1].last_valid_index()])[0:10] <= self.last_date:
            if month_last_6 <= self.last_date[0:7]:
                price_rel_6mo = float(ret_list[1].adjClose[ret_list[1].last_valid_index()])/float(spy_ret_list[1].adjClose[spy_ret_list[1].last_valid_index()])
                ret.append((float(price_rel_6mo) - float(price_rel_1))/float(price_rel_1))
            else:
                ret.append('None')

            #if np.str(ret_list[2].date[ret_list[2].last_valid_index()])[0:10] <= self.last_date:
            if month_last_12 <= self.last_date[0:7]:
                price_rel_12mo = float(ret_list[2].adjClose[ret_list[2].last_valid_index()])/float(spy_ret_list[2].adjClose[spy_ret_list[2].last_valid_index()])
                ret.append((float(price_rel_12mo) - float(price_rel_1))/float(price_rel_1))
            else:
                ret.append('None')
        except:
            return ['None','None','None']

        return ret

    def find_empty(self):
        el = []
        for i, c in enumerate(self.company_list.SYMBOL):
            cpath = join(FORMS_DATA_PATH, c, c, 'form4')
            file_list = [f for f in listdir(cpath)]
            ef = open('empty_folders.txt','a+')

            if not file_list:
                el.append(c)
                ef.write(c + '\n')
        print 'Number of empty folders: ' + str(len(el))
                

if __name__ == '__main__':
    s = SecCrawler()
    s.parse_form4('2010-01-01', '2016-12-31',returns=True)

