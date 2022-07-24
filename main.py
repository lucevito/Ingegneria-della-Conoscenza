from anime_processing import *
from anime_prompter import *

print('Loading data...')
anime_data = pd.read_csv('./anime.csv')
rating_data = pd.read_csv('./rating.csv')

print('Processing data....')
genres_list = genre_extraction(anime_data)
anime_data = anime_data_processing(anime_data)
anime_index = anime_index_extraction(anime_data)
anime_pivot = anime_pivot_processing(anime_data, rating_data)
del rating_data


while True:
    print('Enter 1 if you want to be recommended anime with Collaborative Filtering ')
    print('and Content based filtering by entering the name of an anime you know')
    print('Enter 2 if you want to be recommended an anime by entering the genres you prefer the most')
    print('Enter 0 if you want close the program')
    print('If you enter something else this information will be repeat')
    print('\n')
    recommended_choose = input('Enter your choice : ')

    if recommended_choose == '1':
        print('Do you want proceed with a name test : Shingeki no Kyojin a.k.a. Attack of titan ?')
        title_test_choose = input('Enter y for a yes, something else for no : ')
        if title_test_choose == 'y':
            title = 'Shingeki no Kyojin'
            anime_similarity(anime_pivot,anime_data,genres_list,anime_index,title)
        else:
            while True:
                while True:
                    title = input('Enter name of a anime and check if there are similar name : ')
                    contain_title = anime_pivot[anime_pivot.index.str.contains(title)]
                    if(not contain_title.empty):
                        print(pd.DataFrame(contain_title.index))
                        break
                    else:
                        print('There is no anime with this name')
                title = input('Enter name of a anime: ')
                if title in anime_pivot.index.values :
                    anime_similarity(anime_pivot,anime_data,genres_list,anime_index,title)
                    del title
                    break
                else:
                    print('The name of the anime you entered is missing or misspelled\n')

    elif recommended_choose == '2':

        print('Warning : to reduce the use of ram, the database size will be reduced')
        print('based on the type of anime such as tv movies etc ..')
        while True:
            print('\nEnter 1 for seach with type Movie')
            print('Enter 2 for seach with type Music')
            print('Enter 3 for seach with type ONA')
            print('Enter 4 for seach with type OVA')
            print('Enter 5 for seach with type Special')
            print('Enter 6 for seach with type TV')
            type_select = input('Enter your choice : ')
            if type_select == '1':
                anime_select = anime_data.loc[anime_data['type'] == 'Movie']
                break
            elif type_select == '2':
                anime_select = anime_data.loc[anime_data['type'] == 'Music']
                break
            elif type_select == '3':
                anime_select = anime_data.loc[anime_data['type'] == 'ONA']
                break
            elif type_select == '4':
                anime_select = anime_data.loc[anime_data['type'] == 'OVA']
                break
            elif type_select == '5':
                anime_select = anime_data.loc[anime_data['type'] == 'Special']
                break
            elif type_select == '6':
                anime_select = anime_data.loc[anime_data['type'] == 'TV']
                break

        test = anime_select[0:0]
        test.loc[len(test.index)] = [0,'0','0',0,0,0,  
                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ]
        print('Select one or more genres')
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
        print(' 44 test with genres of Shingeki no Kyojin a.k.a. Attack of titan')
        print(' Enter 0 to stop selection, enter 50 if you want close program')

        while True:
            choice = input('Enter your choice : ')
            if  choice == '1':
                test['Action'] = test['Action'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '2':
                test['Adventure'] = test['Adventure'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '3':
                test['Cars'] = test['Cars'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '4':
                test['Comedy'] = test['Comedy'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '5':
                test['Dementia'] = test['Dementia'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '6':
                test['Demons'] = test['Demons'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '7':
                test['Drama'] = test['Drama'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '8':
                test['Ecchi'] = test['Ecchi'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '9':
                test['Fantasy'] = test['Fantasy'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '10':
                test['Game'] = test['Game'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '11':
                test['Harem'] = test['Harem'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '12':
                test['Hentai'] = test['Hentai'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '13':
                test['Historical'] = test['Historical'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '14':
                test['Horror'] = test['Horror'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '15':
                test['Josei'] = test['Josei'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '16':
                test['Kids'] = test['Kids'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '17':
                test['Magic'] = test['Magic'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '18':
                test['Martial Arts'] = test['Martial Arts'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '19':
                test['Mecha'] = test['Mecha'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '20':
                test['Military'] = test['Military'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '21':
                test['Music'] = test['Music'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '22':
                test['Mystery'] = test['Mystery'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '23':
                test['Parody'] = test['Parody'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '24':
                test['Police'] = test['Police'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '25':
                test['Psychological'] = test['Psychological'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '26':
                test['Romance'] = test['Romance'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '27':
                test['Samurai'] = test['Samurai'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '28':
                test['School'] = test['School'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '29':
                test['Sci-Fi'] = test['Sci-Fi'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '30':
                test['Seinen'] = test['Seinen'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '31':
                test['Shoujo'] = test['Shoujo'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '32':
                test['Shoujo Ai'] = test['Shoujo Ai'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '33':
                test['Shounen'] = test['Shounen'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '34':
                test['Shounen Ai'] = test['Shounen Ai'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '35':
                test['Slice of Life'] = test['Slice of Life'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '36':
                test['Space'] = test['Space'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '37':
                test['Sports'] = test['Sports'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '38':
                test['Super Power'] = test['Super Power'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '39':
                test['Supernatural'] = test['Supernatural'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '40':
                test['Thriller'] = test['Thriller'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '41':
                test['Vampire'] = test['Vampire'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '42':
                test['Yaoi'] = test['Yaoi'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '43':
                test['Yuri'] = test['Yuri'].replace(0, 1)
                print('Genre acquired!')
                print('Enter 0 to stop selection, enter 50 if you do not want this type of recommendation')
            elif  choice == '44':
                test = test[0:0]
                test.loc[len(test.index)] = [0,'0','0',0,0,0,  
                                    1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                                        0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0 ]
                anime_genre_like(anime_select,test)
                break
            
            elif  choice == '0':
                anime_genre_like(anime_select,test)        
                break

            elif  choice == '50':
                print('Close...')
                break
            else:
                print('The entered input is invalid, please enter a correct one ...')

    elif recommended_choose == '0':
        print('Closing program')
        break

