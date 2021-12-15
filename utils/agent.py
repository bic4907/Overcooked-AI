def find_index(arr: list, key: str) -> []:
    index_arr = list()

    for i, item in enumerate(arr):
        if item == key:
            index_arr.append(i)

    return index_arr
