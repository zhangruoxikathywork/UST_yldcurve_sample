'''
Overview
-------------
This Python script aims to load monthly UST bond data from 
on WRDS since 1925.
 
Requirements
-------------
Access to the WRDS server and associated databases.

Package versions 
-------------
pandas v1.4.4
wrds v3.1.2
csv v1.0
'''

#* ************************************** */
#* Libraries                              */
#* ************************************** */ 

import pandas as pd
import wrds
# import config
# from pathlib import Path

WRDS_USERNAME = 'zhangruoxikathy'

## Monthly UST bond data columns
# kytreasno, kycrspid, crspid, tcusip, tdatdt, tmatdt, iwhy,
#             tcouprt, tnippy, tvalfc, tfcpdt, ifcpdtf, tfcaldt, tnotice,
#             iymcn, itype, iuniq, itax, iflwr, tbankdt, tstripelig, tfrgntgt,
#             mcaldt, tmbid, tmask, tmnomprc, tmnomprc_flg, tmsourcr, tmaccint,
#             tmretnua, tmyld, tmduratn, tmtotout, tmpubout, tmpcyld, tmretnxs, tmpdint


def pull_crsp(wrds_username=WRDS_USERNAME):
    """
    Pull UST bond returns data from WRDS CRSP.
    """

    sql_query = """
        SELECT iss.KYTREASNO, iss.KYCRSPID, iss.CRSPID, iss.TCUSIP, 
               TO_CHAR(iss.TDATDT, 'YYYYMMDD') AS TDATDT, 
               TO_CHAR(iss.TMATDT, 'YYYYMMDD') AS TMATDT, 
               iss.IWHY, iss.TCOUPRT, iss.TNIPPY, iss.TVALFC, 
               TO_CHAR(iss.TFCPDT, 'YYYYMMDD') AS TFCPDT,
               iss.IFCPDTF, iss.TFCALDT, iss.TNOTICE, iss.IYMCN, iss.ITYPE, iss.IUNIQ,
               iss.ITAX, iss.IFLWR, iss.TBANKDT, iss.TSTRIPELIG, iss.TFRGNTGT,
               TO_CHAR(mth.MCALDT, 'YYYYMMDD') AS MCALDT, 
               mth.TMBID, mth.TMASK, mth.TMNOMPRC, mth.TMNOMPRC_FLG, 
               mth.TMSOURCR, mth.TMACCINT, mth.TMRETNUA, mth.TMYLD, mth.TMDURATN, 
               mth.TMTOTOUT, mth.TMPUBOUT, mth.TMPCYLD, mth.TMRETNXS, mth.TMPDINT,
               mth.TMIDXRATIO, mth.TMIDXRATIO_FLG
        FROM CRSPM.TFZ_ISS AS iss
        INNER JOIN CRSPM.TFZ_MTH AS mth ON iss.KYTREASNO = mth.KYTREASNO AND iss.KYCRSPID = mth.KYCRSPID;
    """

    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     comp = db.raw_sql(sql_query, date_cols=["date"])
    db = wrds.Connection(wrds_username=wrds_username)
    bond = db.raw_sql(sql_query)
    db.close()

    return bond


def load_crsp(path):
    # path = '../../data/USTM.csv'
    bond = pd.read_csv(path)
    return bond


def _demo():
    bonddata = load_crsp('../../data/USTM.csv')


if __name__ == "__main__":
    bonddata = pull_crsp(wrds_username=WRDS_USERNAME)
    bonddata.to_csv('../../data/USTM.csv')
