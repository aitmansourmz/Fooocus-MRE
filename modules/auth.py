import json
import hashlib

from os.path import exists


def auth_list_to_dict(auth_list):
    auth_dict = {}
    for auth_data in auth_list:
        if 'user' in auth_data:
            if 'hash' in auth_data:
                auth_dict |= {auth_data['user']: auth_data['hash']}
            elif 'pass' in auth_data:
                auth_dict |= {auth_data['user']: hashlib.sha256(bytes(auth_data['pass'], encoding='utf-8')).hexdigest()}
    return auth_dict


def load_auth_data(filename=None):
    auth_dict = None
    if filename != None and exists(filename):
        with open(filename, encoding='utf-8') as auth_file:
            try:
                auth_obj = json.load(auth_file)
                if isinstance(auth_obj, list) and len(auth_obj) > 0:
                    auth_dict = auth_list_to_dict(auth_obj)
            except Exception as e:
                print('load_auth_data, e: ' + str(e))
            finally:
                auth_file.close()
    return auth_dict


auth_dict = load_auth_data('auth.json')

auth_enabled = auth_dict != None


def check_auth(user, password):
    if user not in auth_dict:
        return False
    else:   
        return hashlib.sha256(bytes(password, encoding='utf-8')).hexdigest() == auth_dict[user]
