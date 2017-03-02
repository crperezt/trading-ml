import time
import threading
from sec_crawler_mt import SecCrawler
import pandas as pd

def get_filings(from_company = 0, to_company = 0):
    t1 = time.time()

    # create object
    seccrawler = SecCrawler()

    #TODO - loop to go through company list, cik
    # add a "to date" to the filing_4 function
    # 
    company_list = pd.read_csv("company_list.csv")

    if from_company == 0:
        for i, c in enumerate(company_list.SYMBOL[461:len(company_list.SYMBOL)]):
        #for i, c in enumerate(company_list.SYMBOL):
        #for i, c in enumerate(company_list.SYMBOL):
            companyCode = c    # company code 
            cik = c      # cik code for apple
            to_date = '20161231'       # date to which filings should be downloaded
            from_date = '20100101'

            seccrawler.filing_4(str(companyCode), str(cik), str(from_date), str(to_date))
    else:
        for i, c in enumerate(company_list.SYMBOL[from_company:to_company]):
        #for i, c in enumerate(company_list.SYMBOL):
        #for i, c in enumerate(company_list.SYMBOL):
            companyCode = c    # company code 
            cik = c      # cik code for apple
            to_date = '20161231'       # date to which filings should be downloaded
            from_date = '20100101'

            seccrawler.filing_4(str(companyCode), str(cik), str(from_date), str(to_date))


    t2 = time.time()
    print ("Total Time taken: "),
    print (t2-t1)


class myThread (threading.Thread):
    def __init__(self, threadID, name, from_company, to_company):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.from_company = from_company
        self.to_company = to_company
    def run(self):
        print "Starting " + self.name
        get_filings(self.from_company, self.to_company)
        print "Exiting " + self.name


if __name__ == '__main__':

    # thread_array = []
    # j = 0
    # try:
    #     for i in range(205,2245,680):
    #         thread_array.append(myThread(j, "thread" + str(j), i, i+679))
    #         j = j + 1
    # except:
    #     print "Error: unable to start thread"

    # for t in thread_array:
    #     t.start()

    t0 = myThread(0, "thread0", 580, 885)
    t1 = myThread(1, "thread1", 1246, 1565)
    t2 = myThread(2, "thread2", 1998, 2444)

    t0.start()
    t1.start()
    t2.start()

    print "Exiting Main Thread"