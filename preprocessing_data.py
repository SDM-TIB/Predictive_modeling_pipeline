import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'



def define_class(class0,target):
    target.rename(columns={ target.columns[0]: "first_col", target.columns[1]: "second_col"}, inplace = True)
    id_0 = target.loc[(target.second_col == class0)].first_col.values
    target.loc[(target.first_col.isin(id_0)), 'class'] = 0
    target.loc[~(target.first_col.isin(id_0)), 'class'] = 1
    target = target[['first_col', 'class']]
    target.drop_duplicates(keep='first', inplace=True)
    target.reset_index(drop=True, inplace=True)
    target['class'] = target['class'].astype(int)
    return target


def transform_to_binary(data, attribute, val_a, val_b):
    data.loc[data[attribute]==val_a, attribute]= 0
    data.loc[data[attribute]==val_b, attribute]= 1
    return data.rename(columns={attribute: attribute+'_'+val_b})



def hot_encode(data):
    data.rename(columns={ data.columns[0]: "first_col"}, inplace = True)
    cols = (data.columns).tolist()
    col_list = []
    count = data.T.apply(lambda x: x.nunique(), axis=1)
    for col_name,v in count.items():
        if v == 2:
            col_val = data[col_name].values.ravel()
            unique = pd.unique(col_val)
            val_a, val_b = ["".join(item) for item in unique.astype(str)]
            data = transform_to_binary(data,col_name,val_a,val_b)
        else:
            if col_name != 'first_col':
                col_list.append(col_name)
    new_data = pd.get_dummies(data=data, columns=col_list)
    new_data.drop_duplicates(subset=['first_col'], keep='first', inplace=True)
    new_data.reset_index(drop=True, inplace=True)
    return new_data



def load_data(data,target,class0):
    print("--------- Preprocessing Data --------------")
    data.drop_duplicates(keep='first', inplace=True)
    target.drop_duplicates(keep='first', inplace=True)
    encode_target = define_class(class0,target)
    encode_data = hot_encode(data)
    return encode_data,encode_target






