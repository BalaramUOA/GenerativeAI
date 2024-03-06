def analyze_nps_scores(nps_scores):
    nps_sentiments = []
    for score in nps_scores:
        if score <= 6:
            nps_sentiments.append('Detractor')
        elif score >= 9:
            nps_sentiments.append('Promoter')
        else:
            nps_sentiments.append('Passives')
    return nps_sentiments

