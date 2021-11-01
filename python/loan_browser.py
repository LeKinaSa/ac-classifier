from sys import argv
import pandas as pd
from pathlib import Path
from os import chdir


def main():
    try:
        loan_id = int(argv[1])
    except Exception:
        loan_id = int(input("Loan ID:> "))

    account_df = pd.read_csv('../data/account.csv', sep=';')

    card_df = pd.concat([
        pd.read_csv('../data/card_train.csv', sep=';'),
        pd.read_csv('../data/card_test.csv', sep=';'),
    ])

    client_df = pd.read_csv('../data/client.csv', sep=';')

    disp_df = pd.read_csv('../data/disp.csv', sep=';')

    district_df = pd.read_csv('../data/district.csv', sep=';')
    
    loan_df = pd.concat([
        pd.read_csv('../data/loan_train.csv', sep=';'),
        pd.read_csv('../data/loan_test.csv', sep=';'),
    ])

    trans_df = pd.concat([
        pd.read_csv('../data/trans_train.csv', sep=';'),
        pd.read_csv('../data/trans_test.csv', sep=';'),
    ])

    loan_entry = loan_df[loan_df['loan_id'] == loan_id]

    if len(loan_entry) == 0:
        print("Loan entry not found. Are you sure you entered the right ID?")
        print("Exiting early...")
        exit()

    print("=================")
    print("Loan entry found:")
    print(loan_entry)

    account_id = int(loan_entry.get("account_id"))
    account_entry = account_df[account_df['account_id'] == account_id]

    print("====================")
    print("Account entry found:")
    print(account_entry)

    transactions = trans_df[trans_df['account_id'] == account_id]

    if len(transactions) == 0:
        print("==========================")
        print("No transactions were found")
    else:
        print("==============================:")
        print("Transaction entries were found:")
        print(transactions)
    
    district_id = int(account_entry.get("district_id"))

    print("=========================")
    print("Account's district entry:")
    print(district_df[district_df['code '] == district_id].T)

    dispositions = disp_df[disp_df['account_id'] == account_id]

    print("==========================")
    print(f"Found {len(dispositions)} disposition {'entry' if len(dispositions) == 1 else 'entries'}:")
    print(dispositions)

    disposition_ids = dispositions["disp_id"].to_list()
    credit_cards = card_df[card_df['disp_id'].isin(disposition_ids)]

    if len(credit_cards) == 0:
        print("==========================")
        print("No credit cards were found")
    else:
        print("========================")
        print("Credit cards were found:")
        print(credit_cards)
    
    client_ids = dispositions["client_id"].to_list()
    clients = client_df[client_df['client_id'].isin(client_ids)]

    print("===============")
    print("Client entries:")
    print(clients)

if __name__ == "__main__":
    chdir(Path(__file__).parent)
    main()