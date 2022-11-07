import pandas as pd

#Function that print an anime details
def checkanimegenre(anime_data):
    animename = input('Enter anime name : ')

    if animename in anime_data['name'].values:
        features = anime_data.columns[6:-1]
        print('Genre : ')
        for i in range(len(features.to_list())):
            if anime_data.at[ anime_data[anime_data['name'] == animename].index[0] ,features.to_list()[i]] == 1:
                print(features.to_list()[i])
        print('')
    else:
        print('There is no anime with that name in the dataset')

#Function that check one anime genre
def checkanime1genre(anime_data):
    animename = input('Enter anime name : ')
    if animename in anime_data['name'].values:
        features = anime_data.columns[6:-1]
        while True:
            print('Select one genres')
            print(' 1 Action\n 2 Adventure\n 3 Cars\n 4 Comedy\n 5 Dementia')
            print(' 6 Demons\n 7 Drama\n 8 Ecchi\n 9 Fantasy')
            print(' 10 Game\n 11 Harem\n 12 Hentai\n 13 Historical\n 14 Horror')
            print(' 15 Josei\n 16 Kids\n 17 Magic\n 18 Martial Arts')
            print(' 19 Mecha\n 20 Military\n 21 Music\n 22 Mystery\n 23 Parody')
            print(' 24 Police\n 25 Psychological\n 26 Romance\n 27 Samurai')
            print(' 28 School\n 29 Sci-Fi\n 30 Seinen\n 31 Shoujo\n 32 Shoujo Ai')
            print(' 33 Shounen\n 34 Shounen Ai\n 35 Slice of Life')
            print(' 36 Space\n 37 Sports\n 38 Super Power\n 39 Supernatural')
            print(' 40 Thriller\n 41 Vampire\n 42 Yaoi\n 43 Yuri')
            print(' if you enter a number that does not match a listed gender, the question will be repeated')
            choice = input('Enter your choice : ')
            if int(choice) > 0 and int(choice) < 44:
                break
        if anime_data.at[ anime_data[anime_data['name'] == animename].index[0] ,features.to_list()[int(choice)]] == 1:
            print('Yes\n')
        else:
            print('No\n')
    else:
        print('There is no anime with that name in the dataset\n')

#Function that print yes if 2 anime have common genres (at least one)
def comparisonanime(anime_data):
    features = anime_data.columns[6:-1]
    exit = False
    animename1 = input('Enter first anime name : ')
    if animename1 in anime_data['name'].values:
        animename2 = input('Enter second anime name : ')
        if animename2 in anime_data['name'].values:
            for i in range(len(features.to_list())):
                for j in range(len(features.to_list())):
                    if anime_data.at[ anime_data[anime_data['name'] == animename1].index[0] ,features.to_list()[i]] == 1:
                        if anime_data.at[ anime_data[anime_data['name'] == animename1].index[0] ,features.to_list()[i]] == anime_data.at[ anime_data[anime_data['name'] == animename2].index[0] ,features.to_list()[j]]:
                            print('Yes\n')
                            exit = True
                            break
                if exit == True:
                    break  
        else:
            print('There is no anime with that name in the dataset\n')
    else:
        print('There is no anime with that name in the dataset\n')

    if exit == False:
        print('No\n')

#Function that print an anime details
def animedetails(anime_data):
    animename = input('Enter anime name as example Gintama : ')
    if animename in anime_data['name'].values:
        print(pd.DataFrame({'Anime name': anime_data['name'].loc[anime_data['name'] == animename].values,
                            'Rating': anime_data['rating'].loc[anime_data['name'] == animename].values,
                            'Type': anime_data['type'].loc[anime_data['name'] == animename].values}))

    else:
        print('There is no anime with that name in the dataset\n')

