import pandas as pd
import streamlit as st
from datetime import datetime, time, timedelta
import psycopg2

def convert_float_to_hhmm(hours):
    try:
        hours = float(hours)
        total_minutes = int(round(hours * 60))
        hh = total_minutes // 60
        mm = total_minutes % 60
        return f"{hh:02}:{mm:02}"
    except (ValueError, TypeError):
        return ''

def summarize_engineer_day(raw_df):
    # Ensure datetime columns are properly parsed
    for col in ['Travel Start', 'On Site', 'Off Site']:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], dayfirst=True, errors='coerce')

    raw_df['Date'] = raw_df['Travel Start'].dt.date

    summary_list = []

    for (engineer, date), group in raw_df.groupby(['Engineer', 'Date']):
        all_times = pd.Series(dtype='datetime64[ns]')
        for col in ['Travel Start', 'On Site', 'Off Site']:
            if col in group.columns:
                all_times = pd.concat([all_times, group[col].dropna()])
        all_times = all_times.sort_values()

        travel_start = group['Travel Start'].min()
        on_site = group['On Site'].min()
        arrive_home = all_times.max() if not all_times.empty else pd.NaT
        second_latest = all_times.iloc[-2] if len(all_times) >= 2 else pd.NaT

        summary_list.append({
            'Engineer': engineer,
            'Date': date,
            'Travel Start': travel_start,
            'On Site': on_site,
            'Off Site': second_latest,
            'Arrive Home': arrive_home
        })

    return pd.DataFrame(summary_list)


def exclude_names(df):

    exclude_names = [
        "Alan Cain", "David Allot", "Dean McLennan", "Jayden Williams",
        "Kevin McGuinness", "Regan Bayfield", "Ryan Lawton", "Scott Pinder",
        "Dominic Rathburn", "Ryan Bayfield", "Grant Luke", "Jordan Gleave","Paul Coffey","Admin User"
    ]

    mask = (
            ~df['Engineer'].str.lower().isin([name.lower() for name in exclude_names])
            & ~df['Engineer'].str.lower().str.contains(r"\(s\)", na=False, regex=True)
            & ~df['Engineer'].str.lower().str.contains('office', na=False)
    )

    df = df[mask].reset_index(drop=True)

    return df


def merge_npt_data(productive_df, npt_filepath):
    # Load NPT CSV
    npt_df = pd.read_csv(npt_filepath, dayfirst=True)

    npt_df['StartDate'] = pd.to_datetime(npt_df['Non Productive Start'], dayfirst=False)
    npt_df['EndDate'] = pd.to_datetime(npt_df['Non Productive End'], dayfirst=True)
    npt_df['Start Time'] = npt_df['StartDate'].dt.time
    npt_df['Finish Time'] = npt_df['EndDate'].dt.time

    # Create date key for join
    npt_df['Date'] = npt_df['StartDate'].dt.date
    npt_df.rename(columns={'Engineer Name': 'Engineer'}, inplace=True)
    npt_df['Engineer'] = npt_df['Engineer'].astype(str).str.strip()

    # Format NPT Time: "HH:MM - HH:MM"
    npt_df['Time Window'] = npt_df.apply(
        lambda row: f"{row['Start Time'].strftime('%H:%M')} - {row['Finish Time'].strftime('%H:%M')}"
        if pd.notnull(row['Start Time']) and pd.notnull(row['Finish Time']) else "", axis=1
    )

    st.title("NPT")
    st.dataframe(npt_df)

    # Group and concatenate
    npt_summary = npt_df.groupby(['Engineer', 'Date']).agg({
        'Reason': lambda x: ", ".join(x.dropna().astype(str)),
        'Time Window': lambda x: ", ".join(x.dropna())
    }).reset_index()

    npt_summary.rename(columns={'Time Window': 'NPT Time'}, inplace=True)
    # npt_summary['Reason'] = 'Unpaid'

    st.title("NPT Data")
    st.dataframe(npt_summary)

    # Merge into productive_rate_df
    productive_df['Engineer'] = productive_df['Engineer'].astype(str).str.strip()
    enriched_df = pd.merge(productive_df, npt_summary, on=['Engineer', 'Date'], how='outer')

    # ✅ Fill missing columns with default values
    enriched_df['Reason'] = enriched_df['Reason'].fillna('')  # Reason for NPT
    enriched_df['NPT Time'] = enriched_df['NPT Time'].fillna('')
    return enriched_df

def calculate_npt_paid_unpaid_hours(df):
    unpaid_keywords = ['unpaid rest', 'unpaid hours']

    def parse_npt_slots(reason_str, time_window_str):
        if not reason_str or not time_window_str:
            return 0.0, 0.0

        # Pre-split
        reason_parts = [r.strip().lower() for r in reason_str.split(",")]
        time_slots = [t.strip() for t in time_window_str.split(",")]

        # 🔍 Check first if ANY time slot has unpaid-related reason
        if not any(any(keyword in reason for keyword in unpaid_keywords) for reason in reason_parts):
            return 0.0, 0.0  # ⛔ Skip processing this row

        unpaid_total = 0.0
        paid_total = 0.0

        for reason, time_slot in zip(reason_parts, time_slots):
            try:
                start_str, end_str = time_slot.split(" - ")
                start = datetime.strptime(start_str, "%H:%M")
                end = datetime.strptime(end_str, "%H:%M")

                duration = (end - start).total_seconds() / 3600
                if duration < 0:
                    duration += 24  # handle wraparound

                if any(keyword in reason for keyword in unpaid_keywords):
                    unpaid_total += duration
                else:
                    paid_total += duration
            except:
                continue  # skip malformed rows

        return round(paid_total, 2), round(unpaid_total, 2)

    # Apply only to rows that contain "unpaid" in their Reason column
    df[['NPT Paid Hours with reason', 'NPT Unpaid Hours']] = df.apply(
        lambda row: pd.Series(parse_npt_slots(row.get('Reason', ''), row.get('NPT Time', ''))),
        axis=1
    )
    return df

# Define a function to calculate total hours
def calculate_total_hours(row):
    if pd.notnull(row['Travel Start']) and pd.notnull(row['Arrive Home']):
        return round((row['Arrive Home'] - row['Travel Start']).total_seconds() / 3600, 2)
    elif all(pd.isnull(row[col]) for col in ['Travel Start', 'On Site', 'Off Site', 'Arrive Home']):
        reason = str(row.get('Reason', '')).lower()
        if 'unpaid rest' not in reason and 'unpaid hours' not in reason:
            return 8

def calculate_total_standard_work_hours(productive_NPT_df):
    # Convert misleading float (e.g., 2.45 meaning 2h 45m) to actual hours
    def misleading_decimal_to_hours(val):
        if pd.isnull(val):
            return 0.0
        try:
            val = float(val)
            hours = int(val)
            minutes = round((val - hours) * 100)  # get 45 from .45
            return round(hours + minutes / 60.0, 2)
        except:
            return 0.0

    # Apply conversion
    productive_NPT_df['Travel Hours'] = productive_NPT_df['Travel HH:MM'].apply(misleading_decimal_to_hours)
    productive_NPT_df['Labour Hours'] = productive_NPT_df['Labour HH:MM'].apply(misleading_decimal_to_hours)

    # Total standard hours
    productive_NPT_df['Total Standard Hours'] = productive_NPT_df['Travel Hours'] + productive_NPT_df['Labour Hours']
    productive_NPT_df['Total Travel Time'] = productive_NPT_df['Travel Hours']

    # Group by Engineer and Date
    grouped = productive_NPT_df.groupby(['Engineer', 'Date'], as_index=False).agg({
        'Total Standard Hours': 'sum',
        'Total Travel Time': 'sum'
    })

    # Fill missing with 0
    grouped['Total Standard Hours'] = grouped['Total Standard Hours'].fillna(0).round(2)
    grouped['Total Travel Time'] = grouped['Total Travel Time'].fillna(0).round(2)

    return grouped


def get_time_sheet_raw():
    conn = psycopg2.connect(
        host="fc-hub-ee50ee54-d079e7a3bad741b4b4e2190db25c7856.postgres.database.azure.com",
        port="5432",
        database="postgres",
        user="ro_user",
        password="qum7w26xxy1k5338apl41wm48c1orq",
        sslmode="require"
    )

    # Step 1: Load original timesheet data
    cur = conn.cursor()
    cur.execute("SELECT * FROM timesheet_raw WHERE start >= '2025-06-01' AND start < '2025-07-01';")
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    st.title("Time Sheet Raw")
    st.dataframe(df)

def calculate_travel_start_paid_unpaid(df):
    """
    Adds 'Travel Start Paid' and 'Travel Start Unpaid' columns to the DataFrame
    based on the Travel Start time and On Site time.
    """
    def classify_row(row):
        travel_start = row.get('Travel Start')
        on_site = row.get('On Site')

        if pd.isnull(travel_start) or pd.isnull(on_site):
            return pd.Series([0.0, 0.0])

        duration = round((on_site - travel_start).total_seconds() / 3600, 2)
        if duration < 0:
            return pd.Series([0.0, 0.0])

        t = travel_start.time()

        if t < time(5, 0):
            return pd.Series([round(duration * 2, 2), 0.0])
        elif t < time(6, 0):
            return pd.Series([round(duration * 1.5, 2), 0.0])
        elif t < time(7, 30):
            return pd.Series([round(duration, 2), 0.0])
        else:
            return pd.Series([0.0, 0.5])  # flat 30 min unpaid

    df[['Travel Start Paid', 'Travel Start Unpaid']] = df.apply(classify_row, axis=1)
    return df

def calculate_lunch_deductions(df):
    """
    Adds 'Lunch Deduction (hrs)' column:
    - 0.0 if Adjusted Total Hours < 4
    - 0.5 if ≥ 4
    """
    df['Lunch Deduction (hrs)'] = df['Adjusted Total Hours'].apply(
        lambda x: 0.0 if pd.isnull(x) or x < 4 else 0.5
    )
    return df

def get_rate_category(day: str, current_time: datetime, base_date: datetime) -> str:
    """
    Determines rate category based on day of week and time.
    """
    if day.lower() == 'sunday':
        return 'R2'
    elif day.lower() == 'saturday':
        return 'R1.5' if current_time.time() < time(12, 0) else 'R2'
    elif current_time.date() > base_date.date() and current_time.time() < time(6, 0):
        return 'R2'
    elif current_time.time() < time(6, 0):
        return 'R1.5'
    elif current_time.time() >= time(17, 0):
        return 'R1.5'
    else:
        return 'R1'

def calculate_travel_end_deductions(row):
    """
    Deducts up to 30 minutes for the last travel of the day.
    Deduct only the actual duration if it's < 30 mins.
    Deduct at the rate applicable at the 'Arrive Home' time.

    Returns the deduction in hours:
    - 0.5, 0.75, or 1.0 based on rate (for full 30 mins)
    - or proportionally smaller if travel was shorter.
    """
    date = pd.to_datetime(row.get('Date'), errors='coerce')
    day = row.get('Day')
    arrive_home = row.get('Arrive Home')
    off_site = row.get('Off Site')

    if pd.isnull(date) or pd.isnull(arrive_home) or pd.isnull(off_site) or not isinstance(day, str):
        return 0.0

    # Calculate travel duration (from Off Site to Arrive Home)
    travel_duration_secs = (arrive_home - off_site).total_seconds()
    if travel_duration_secs <= 0:
        return 0.0

    travel_duration_mins = travel_duration_secs / 60
    travel_deduct_mins = min(travel_duration_mins, 30)  # Cap at 30 mins
    travel_deduct_hrs = round(travel_deduct_mins / 60, 2)

    # Get rate category based on arrival time
    deduction_time = datetime.combine(date, arrive_home.time())
    rate = get_rate_category(day, deduction_time, datetime.combine(date, time(0, 0)))

    if rate == 'R1':
        return round(travel_deduct_hrs * 1, 2)
    elif rate == 'R1.5':
        return round(travel_deduct_hrs * 1.5, 2)
    elif rate == 'R2':
        return round(travel_deduct_hrs * 2, 2)
    else:
        return round(travel_deduct_hrs, 2)

def calculate_site_time(df):
    """
    Adds a 'Site Time (hrs)' column as float.
    """
    def compute(row):
        on_site = row.get('On Site')
        off_site = row.get('Off Site')

        if pd.notnull(on_site) and pd.notnull(off_site):
            duration_td = off_site - on_site
            return round(duration_td.total_seconds() / 3600, 2)
        return 0.0

    df['Site Time (hrs)'] = df.apply(compute, axis=1)
    return df


def calculate_total_travel_time(df):
    def compute(row):
        try:
            morning = (
                (row['On Site'] - row['Travel Start']).total_seconds() / 3600
                if pd.notnull(row['On Site']) and pd.notnull(row['Travel Start']) else 0.0
            )
            evening = (
                (row['Arrive Home'] - row['Off Site']).total_seconds() / 3600
                if pd.notnull(row['Arrive Home']) and pd.notnull(row['Off Site']) else 0.0
            )
            total = morning + evening
            return round(max(total, 0), 2)  # Prevent negatives
        except:
            return 0.0

    df['Total Travel Time'] = df.apply(compute, axis=1)
    return df

def calculate_explicit_npt_paid_hours(df):
    def compute(row):
        # Only if core times are present and NPT Time + Reason are provided
        if (
                pd.notnull(row.get('Travel Start')) and
                pd.notnull(row.get('On Site')) and
                pd.notnull(row.get('Off Site')) and
                pd.notnull(row.get('Arrive Home')) and
                pd.notnull(row.get('Reason')) and
                pd.notnull(row.get('NPT Time'))
        ):
            reason = str(row['Reason']).lower()
            if 'unpaid' in reason:
                return 0.0  # skip if unpaid reason is involved

            try:
                start_str, end_str = row['NPT Time'].split('-')
                start = datetime.strptime(start_str.strip(), "%H:%M")
                end = datetime.strptime(end_str.strip(), "%H:%M")
                duration = (end - start).total_seconds() / 3600
                if duration < 0:
                    duration += 24
                return round(duration, 2)
            except:
                return 0.0
        else:
            return 0.0

    df['NPT Paid Hours'] = df.apply(compute, axis=1)
    return df

def allocate_pay_rates(row):
    total = row.get('Remaining Payable Hours', 0)
    arrive_home = row.get('Arrive Home')
    day = str(row.get('Day', '')).lower()

    if pd.isnull(arrive_home) or total <= 0:
        return pd.Series([0.0, 0.0, 0.0], index=['R1 (hrs)', 'R1.5 (hrs)', 'R2 (hrs)'])

    # If total is less than or equal to 8, assign all to R1
    if total <= 8:
        return pd.Series([round(total, 2), 0.0, 0.0], index=['R1 (hrs)', 'R1.5 (hrs)', 'R2 (hrs)'])

    # Else, assign 8 to R1, remaining to the appropriate rate
    r1 = 8.0
    remaining = round(total - r1, 2)

    # Get rate category at the end of the shift
    rate_time = datetime.combine(arrive_home.date(), arrive_home.time())
    base_date = datetime.combine(arrive_home.date(), time(0, 0))
    rate = get_rate_category(day, rate_time, base_date)

    r15, r2 = 0.0, 0.0
    if rate == 'R1.5':
        r15 = remaining
    elif rate == 'R2':
        r2 = remaining

    return pd.Series([r1, round(r15, 2), round(r2, 2)], index=['R1 (hrs)', 'R1.5 (hrs)', 'R2 (hrs)'])

def override_final_hours_for_empty_shifts(row):
    if all(pd.isnull(row.get(col)) for col in ['Travel Start', 'On Site', 'Off Site', 'Arrive Home']) and \
             row.get('NPT Time') and row.get('Reason'):
        return row.get('Total Hours', 0.0)
    return row.get('Final Payable Hours', 0.0)

if __name__ == "__main__":
    raw_df = pd.read_csv("Timesheet_June2025_updated.csv", encoding='utf-8-sig')
    summary_df = summarize_engineer_day(raw_df)

    summary_df = exclude_names(summary_df)
    productive_NPT_df = merge_npt_data(summary_df, "NPT_June2025_updated.csv")
    # productive_NPT_df['Date'] = productive_NPT_df['Date'].astype(str)
    productive_NPT_df = calculate_npt_paid_unpaid_hours(productive_NPT_df)

    # Merge with date_dimension
    date_dim = pd.read_csv('date_dimension.csv', dayfirst=True)
    date_dim['Date'] = pd.to_datetime(date_dim['Date'], dayfirst=True).dt.date
    productive_NPT_df = pd.merge(productive_NPT_df, date_dim[['Date', 'Day']], on='Date', how='left')

    # Apply the function
    productive_NPT_df['Total Hours'] = productive_NPT_df.apply(calculate_total_hours, axis=1)

    productive_NPT_df['Adjusted Total Hours'] = (
            productive_NPT_df['Total Hours'].fillna(0) +
            productive_NPT_df['NPT Paid Hours with reason'].fillna(0) -
            productive_NPT_df['NPT Unpaid Hours'].fillna(0)
    ).clip(lower=0).round(2)

    standard_work_summary_df = calculate_total_standard_work_hours(raw_df)

    # Ensure Date types match (both should be strings or datetime)
    productive_NPT_df['Date'] = productive_NPT_df['Date'].astype(str)
    standard_work_summary_df['Date'] = standard_work_summary_df['Date'].astype(str)

    # Merge both DataFrames
    merged_df = pd.merge(
        productive_NPT_df,
        standard_work_summary_df,
        on=['Engineer', 'Date'],
        how='left'
    )

    # Fill missing standard columns with default values
    merged_df['Total Standard Hours'] = merged_df['Total Standard Hours'].fillna(0).round(2)

    merged_df['Standard Hrs Diff'] = (
            merged_df['Adjusted Total Hours']
            - merged_df['Total Standard Hours']
    ).round(2)

    # 🔁 Adjust Paid Hours by Subtracting Gap Hours
    merged_df['Total Hours without gaps'] = (
            merged_df['Adjusted Total Hours'] - merged_df['Standard Hrs Diff']
    ).apply(lambda x: round(max(x, 0), 2))  # Ensure no negative values

    merged_df = calculate_explicit_npt_paid_hours(merged_df)
    merged_df['Total Payable Hours'] = merged_df['Total Hours without gaps'] + merged_df['NPT Paid Hours']

    merged_df = calculate_travel_start_paid_unpaid(merged_df)

    merged_df = calculate_lunch_deductions(merged_df)

    merged_df['Travel End Deduction'] = merged_df.apply(calculate_travel_end_deductions, axis=1)
    merged_df = calculate_site_time(merged_df)

    # merged_df = calculate_site_time(merged_df)
    st.title("Total Site Time")
    st.dataframe(merged_df)
    #merged_df = calculate_total_travel_time(merged_df)
    merged_df['Remaining Payable Hours'] = (merged_df['Total Payable Hours'] - merged_df['Total Travel Time'] - merged_df['Lunch Deduction (hrs)']).apply(lambda x: round(max(x, 0), 2))

    merged_df[['R1 (hrs)', 'R1.5 (hrs)', 'R2 (hrs)']] = merged_df.apply(allocate_pay_rates, axis=1)

    merged_df['Total Payable Onsite Hours'] = (merged_df['R1 (hrs)'] + (merged_df['R1.5 (hrs)'] * 1.5) + (merged_df['R2 (hrs)'] * 2 )).apply(lambda x: round(max(x, 0), 2))
    merged_df['Total Payable Travel and Lunch Hours'] = (merged_df['Travel Start Paid'] - merged_df['Travel Start Unpaid'] - merged_df['Lunch Deduction (hrs)'] - merged_df['Travel End Deduction']).apply(lambda x: round(max(x, 0), 2))

    merged_df['Final Payable Hours'] = (merged_df['R1 (hrs)'] +
           (merged_df['R1.5 (hrs)'] * 1.5) +
            (merged_df['R2 (hrs)'] * 2 )+
            merged_df['Travel Start Paid'] -
            merged_df['Travel Start Unpaid'] -
            merged_df['Lunch Deduction (hrs)'] +
            merged_df['Travel End Deduction']
    ).apply(lambda x: round(max(x, 0), 2))

    merged_df['Final Payable Hours'] = merged_df.apply(override_final_hours_for_empty_shifts, axis=1)

    #Apply HH:MM formatting to key float-based time columns
    time_columns = [
        'Total Hours', 'Adjusted Total Hours', 'Total Standard Hours', 'Standard Hrs Diff', 'Total Hours without gaps',
        'NPT Paid Hours', 'Total Payable Hours', 'Travel Start Paid',
        'Travel Start Unpaid', 'Lunch Deduction (hrs)', 'Travel End Deduction',
        'Remaining Payable Hours', 'Site Time (hrs)',
        'Total Travel Time', 'R1 (hrs)', 'R1.5 (hrs)', 'R2 (hrs)',
        'Total Payable Onsite Hours', 'Total Payable Travel and Lunch Hours', 'Final Payable Hours'
    ]

    for col in time_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')  # convert safely to float
        hhmm_col = f"{col}"
        merged_df[hhmm_col] = merged_df[col].apply(convert_float_to_hhmm)

    st.title("Final df")
    st.dataframe(merged_df)