import pandas as pd
import numpy as np

def load_mbfc():
    mbfc_raw = pd.read_json("./mbfc/sources-pretty.json")
    mbfc = mbfc_raw[["domain", "bias", "credibility", "reporting"]].copy()

    credibility_mapping = {"low-credibility": -1,
                           "medium-credibility": 0,
                           "high-credibility": 1}
    mbfc["credibility_score"] = mbfc["credibility"].apply(lambda x: credibility_mapping.get(x, None))

    reporting_mapping = {"very-low": -2.5,
                         "low": -1.5,
                         "mixed": -0.5,
                         "mostly-factual": 0.5,
                         "high": 1.5,
                         "very-high": 2.5}
    mbfc["reporting_score"] = mbfc["reporting"].apply(lambda x: reporting_mapping.get(x, None))

    polbias_mapping = {"left": -2,
                       "left-center": -1,
                       "center": 0,
                       "right-center": 1,
                       "right": 2}
    mbfc["polbias"] = mbfc["bias"].apply(lambda x: polbias_mapping.get(x, None))

    fn_domain2polbias = {
        "zerohedge.com": 2,
        "breitbart.com": 2,
        "foxnews.com": 2,
        "theepochtimes.com": 2,
        "theblaze.com": 2,
        "globalresearch.ca": None,
        "rt.com": 1,
        "trialsitenews.com": None,
        "sott.net": 2,
        "swarajyamag.com": 2,
        "wnd.com": 2,
        "hindustantimes.com": -1,
        "news18.com": 1,
        "oann.com": 2,
        "washingtontimes.com": 1,
        "prevention.com": None,
        "thriveglobal.com": -2,
        "townhall.com": 2,
        "newsmax.com": 2,
        "infowars.com": 2,
        "thegatewaypundit.com": 2,
        "americanthinker.com": 2,
        "lifezette.com": 2,
        "gellerreport.com": 2,
        "cbn.com": 2,
        "dailymail.co.uk": 2,
        "tass.com": 1,
        "pjmedia.com": 2,
        "ndtv.com": 1,
        "antiwar.com": 1,
        "thefederalist.com": 2,
        "odysee.com": 2,
        "vdare.com": 2,
        "indiatimes.com": 1,
        "amren.com": 2,
        "ria.ru": 1,
        "rumble.com": 2,
        "news.gab.com": 2,
        "unz.com": 2,
        "arabnews.com": 1,
        "therightscoop.com": 2,
        "voiceofeurope.com": 2,
        "justthenews.com": 2,
        "westernjournal.com": 2,
        "frontpagemag.com": 2,
        "bitchute.com": 2,
        "summit.news": 2,
        "lewrockwell.com": 2,
        "naturalnews.com": 2,
        "dailystar.co.uk": 1,
        "dnyuz.com": 1,
    }

    mbfc["polbias"] = mbfc.apply(lambda x: fn_domain2polbias.get(x["domain"], None) if x["bias"] in ["fake-news", "conspiracy-pseudoscience"] else x["polbias"], axis=1)

    #update wrong newsbreak record https://mediabiasfactcheck.com/news-break/
    mbfc["bias"] = mbfc.apply(lambda x: "left-center" if x["domain"] == "newsbreak.com" else x["bias"], axis=1)
    mbfc["polbias"] = mbfc.apply(lambda x: -1 if x["domain"] == "newsbreak.com" else x["polbias"], axis=1)
    mbfc["credibility_score"] = mbfc.apply(lambda x: 0 if x["domain"] == "newsbreak.com" else x["credibility_score"], axis=1)
    mbfc["reporting_score"] = mbfc.apply(lambda x: -0.5 if x["domain"] == "newsbreak.com" else x["reporting_score"], axis=1)

    mbfc = mbfc[["domain", "bias", "credibility_score", "reporting_score", "polbias"]]

    mbfc.loc[len(mbfc)] = ["sciencemag.org", "pro-science", 1.0, 2.5, np.nan] #copied from science.org
    mbfc.loc[len(mbfc)] = ["bbc.co.uk", "left-center", 1.0, 1.5, -1.] #copied from bbc.com
    mbfc.loc[len(mbfc)] = ["yahoo.com", "left-center", 1.0, 1.5, -1.] #copied from news.yahoo.com
    mbfc.loc[len(mbfc)] = ["sputniknews.com", "fake-news", -1., -2., 1.] #copied from sputnikglobe.com
    mbfc.loc[len(mbfc)] = ["pamelageller.com", "fake-news", -1., -1.5, 2.] #copied from gellerreport.com
    mbfc.loc[len(mbfc)] = ["timesofindia.com", "right-center", 0., -0.5, 1.] #copied from indiatimes.com
    mbfc.loc[len(mbfc)] = ["tass.ru", "fake-news", -1., -0.5, 1.] #copied from indiatimes.com
    mbfc.loc[len(mbfc)] = ["cnsnews.com", "fake-news", -1., -1.5, 2.] #copied from indiatimes.com

    mbfc.columns = ["domain", "mbfc_bias", "mbfc_cred", "mbfc_reporting", "mbfc_polbias"]

    return mbfc

mbfc = load_mbfc()

def get_mbfc_domain2bias():
    mbfc_domain2bias = {d:b for d,b in zip(mbfc["domain"].tolist(), mbfc["mbfc_bias"].tolist())}
    return mbfc_domain2bias

def get_mbfc_domain2polbias():
    mbfc_domain2polbias = {d:b for d,b in zip(mbfc["domain"].tolist(), mbfc["mbfc_polbias"].tolist())}
    return mbfc_domain2polbias
