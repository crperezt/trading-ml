import time
from sec_crawler import SecCrawler
import pandas as pd

def get_filings():
    t1 = time.time()

    # create object
    seccrawler = SecCrawler()

    #TODO - loop to go through company list, cik
    # add a "to date" to the filing_4 function
    # 
    company_list = pd.read_csv("empty_folders.txt")
    print company_list
    

    #for i, c in enumerate(company_list.SYMBOL[608:len(company_list.SYMBOL)]):
    #for i, c in enumerate(company_list.SYMBOL):
    for i, c in enumerate(company_list.COMPANY):
        companyCode = c    # company code 
        cik = c      # cik code for apple
        to_date = '20161130'       # date to which filings should be downloaded
        from_date = '20120101'

        seccrawler.filing_4(str(companyCode), str(cik), str(from_date), str(to_date))

    t2 = time.time()
    print ("Total Time taken: "),
    print (t2-t1)

if __name__ == '__main__':
    get_filings() 
