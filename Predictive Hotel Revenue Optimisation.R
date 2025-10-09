## ===========================================
## HOTEL BOOKING DATA ANALYSIS: COMPLETE PIPELINE
## ===========================================

library(randomForest)    # for Random Forest
library(xgboost)         # for XGBoost
library(Matrix)          # for XGBoost
library(pROC)            # ROC-AUC
library(caret)           # confusion matrix, varImp
library(PRROC)           # PR curves
library(ggplot2)         # data visualization
library(gridExtra)       # format ggplots 
library(dplyr)
library(stats)
library(tidyverse)
library(scales) 
library(lubridate)
library(fastDummies)
library(factoextra)
library(rpart)
library(rpart.plot)
library(rattle)
library(car)
library(FactoMineR)

rm(list = ls())
options(stringsAsFactors = FALSE)

## =========================
## DATA CLEANING & PREPROCESSING
## =========================

## Read raw data
df <- read.csv("/Users/lijiarui/Desktop/NTU/Analytics Strategy/Project/hotel_bookings.csv")

## Overview of data
cat("Rows:", nrow(df), " Cols:", ncol(df), "\n")
str(df)
summary(df)

## -----------------------------------
## DATA CLEANING
## -----------------------------------

## Remove duplicated rows
dup_n <- sum(duplicated(df))
cat("Duplicated rows:", dup_n, "\n")
if (dup_n > 0) df <- df[!duplicated(df), ]
dup_n <- sum(duplicated(df))
cat("Duplicated rows after removal:", dup_n, "\n")

## Remove whitespace from character columns
char_cols <- names(df)[sapply(df, is.character)]
for (cn in char_cols) df[[cn]] <- trimws(df[[cn]])

## Convert specific columns to factor
fac_cols <- c("hotel","meal","market_segment","distribution_channel",
              "reserved_room_type","assigned_room_type","deposit_type",
              "customer_type","country","arrival_date_month")
for (cn in intersect(fac_cols, names(df))) df[[cn]] <- factor(df[[cn]])

## Convert is_canceled to integer (0/1)
if ("is_canceled" %in% names(df)) df$is_canceled <- as.integer(df$is_canceled)

## Convert agent/company to integer
num_maybe <- c("agent","company")
for (cn in intersect(num_maybe, names(df))) {
  suppressWarnings( df[[cn]] <- as.integer(df[[cn]]) )
}

## Handle missing values
na_count <- colSums(is.na(df))
na_count[na_count > 0]

## Set missing values to 0 for children and babies
for (cn in c("children","babies")) {
  if (cn %in% names(df)) {
    df[[cn]][is.na(df[[cn]])] <- 0
  }
}

## Fill NA as 0 for agent/company (0 = direct booking)
if ("agent" %in% names(df))   df$agent[is.na(df$agent)] <- 0L
if ("company" %in% names(df)) df$company[is.na(df$company)] <- 0L

## -----------------------------------
## DATA VALIDATION & FEATURE ENGINEERING
## -----------------------------------

## Validate total_guests >= 1; remove invalid rows
if (all(c("adults","children","babies") %in% names(df))) {
  df$total_guests <- with(df, adults + children + babies)
  invalid_guests_idx <- which(df$total_guests <= 0)
  cat("Rows with total_guests <= 0:", length(invalid_guests_idx), "\n")
  df <- df[df$total_guests > 0, ]
}

## Aggregate stays_total_nights
if (all(c("stays_in_weekend_nights","stays_in_week_nights") %in% names(df))) {
  df$stays_total_nights <- with(df, stays_in_weekend_nights + stays_in_week_nights)
}

## Validate ADR (Average Daily Rate) not negative
if ("adr" %in% names(df)) {
  neg_adr_idx <- which(df$adr < 0)
  cat("Rows with negative ADR:", length(neg_adr_idx), "\n")
  df <- df[df$adr >= 0, ]
}

## Create proper date variable
month_levels <- c("January","February","March","April","May","June",
                  "July","August","September","October","November","December")
if ("arrival_date_month" %in% names(df)) {
  df$arrival_date_month <- factor(df$arrival_date_month, levels = month_levels, ordered = TRUE)
}

## Combine date components into arrival_date
make_date_ok <- all(c("arrival_date_year","arrival_date_month","arrival_date_day_of_month") %in% names(df))
if (make_date_ok) {
  mnum <- as.integer(df$arrival_date_month)  ## 1..12
  y <- df$arrival_date_year
  d <- df$arrival_date_day_of_month
  suppressWarnings({
    df$arrival_date <- as.Date(sprintf("%04d-%02d-%02d", y, mnum, d))
  })
}

## Create Value-loss label
if (all(c("reservation_status","deposit_type") %in% names(df))) {
  df$value_loss <- 0L
  df$value_loss[df$reservation_status == "Canceled"] <- 1L
  ns <- df$reservation_status == "No-Show"
  df$value_loss[ ns & df$deposit_type != "Non Refund" ] <- 1L
  df$value_loss <- as.integer(df$value_loss)
  
  cat("\n[Value loss rate overall]:\n")
  print(mean(df$value_loss, na.rm = TRUE))
}

## Create room mismatch flag
if (all(c("reserved_room_type","assigned_room_type") %in% names(df))) {
  rrt_chr <- as.character(df$reserved_room_type)
  art_chr <- as.character(df$assigned_room_type)
  mismatch <- rrt_chr != art_chr
  mismatch[is.na(mismatch)] <- FALSE
  df$room_mismatch <- factor(ifelse(mismatch, "Mismatch", "Match"))
}

## Quick sanity checks for room mismatch
cat("\n[room_mismatch] counts:\n"); print(table(df$room_mismatch))
if ("is_canceled" %in% names(df)) {
  cat("\n[room_mismatch × is_canceled] row %:\n")
  print(prop.table(table(df$room_mismatch, df$is_canceled), 1))
}
if ("is_repeated_guest" %in% names(df)) {
  cat("\n[room_mismatch × is_repeated_guest] row %:\n")
  print(prop.table(table(df$room_mismatch, df$is_repeated_guest), 1))
}

## Cross-tab & statistical tests for room mismatch
if (all(c("room_mismatch","is_repeated_guest") %in% names(df))) {
  cat("\n[room_mismatch × is_repeated_guest] counts:\n")
  tb_mm_rep <- table(df$room_mismatch, df$is_repeated_guest)
  print(tb_mm_rep)
  cat("\n[row %]:\n")
  print(prop.table(tb_mm_rep, 1))
  
  cat("\n[Chi-square test] room_mismatch ~ is_repeated_guest:\n")
  print(chisq.test(tb_mm_rep))
  
  cramers_v <- function(tab) {
    chi <- suppressWarnings(chisq.test(tab, correct = FALSE)$statistic)
    n   <- sum(tab)
    r   <- nrow(tab); c <- ncol(tab)
    as.numeric( sqrt( chi / (n * (min(r, c) - 1)) ) )
  }
  cat("\n[Cramer's V]:\n")
  print(cramers_v(tb_mm_rep))
}

## ================================
## ADDITIONAL DATA PRE-PROCESSING 
## ================================

## Remove unnecessary columns
df <- df[, !(names(df) %in% c("agent", "company", "reservation_status_date"))]

## Check for missing values
na_count <- colSums(is.na(df))
na_count[na_count > 0]

## Convert numeric columns
numeric_cols <- c(
  "lead_time", "arrival_date_year", "arrival_date_week_number",
  "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights",
  "adults", "children", "babies",
  "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
  "days_in_waiting_list", "adr", "required_car_parking_spaces",
  "total_of_special_requests", "total_guests", "stays_total_nights"
)
df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)

## Convert categorical columns
categorical_cols <- c(
  "hotel", "arrival_date_month", "meal", "country", "market_segment",
  "distribution_channel", "reserved_room_type", "assigned_room_type",
  "deposit_type", "customer_type", "reservation_status",
  "is_canceled", "is_repeated_guest"
)
df[categorical_cols] <- lapply(df[categorical_cols], factor)

## Convert date columns
date_cols <- c("arrival_date")
df[date_cols] <- lapply(df[date_cols], function(x) as.Date(x, format="%d-%m-%Y"))

## Reduce country feature cardinality
top_n <- 30
top_countries <- names(sort(table(df$country), decreasing = TRUE))[1:top_n]
df$country <- as.character(df$country)
df$country[!(df$country %in% top_countries)] <- "Other"
df$country <- factor(df$country)

## Save cleaned dataset
df_final_cleaned <- df

## Create value_loss target variable
## Flags if hotel loses money due to cancellation/no-show without deposit
df$value_loss <- with(df, ifelse(
  reservation_status == "Canceled" & deposit_type == "No Deposit", 1,
  ifelse(reservation_status == "Canceled" & deposit_type == "Refundable", 1,
         ifelse(reservation_status == "Canceled" & deposit_type == "Non Refund", 0,
                ifelse(reservation_status == "No-Show" & deposit_type == "No Deposit", 1,
                       ifelse(reservation_status == "No-Show" & deposit_type == "Refundable", 1, 0))))))
df$value_loss <- as.factor(df$value_loss)

## -----------------------------------
## DESCRIPTIVE EXPLORATION
## -----------------------------------

## Overall Cancellation rate
if ("is_canceled" %in% names(df)) {
  cat("\nOverall cancellation rate:\n")
  print( prop.table(table(df$is_canceled)) )
}

## Cancellation rate by hotel type
if (all(c("hotel","is_canceled") %in% names(df))) {
  cat("\nCancellation rate by hotel:\n")
  tab_hotel_cancel <- table(df$hotel, df$is_canceled)
  print(tab_hotel_cancel)
  print( prop.table(tab_hotel_cancel, margin = 1) )  
}

## Cancellation by key factors
for (cn in c("market_segment","distribution_channel","deposit_type","customer_type")) {
  if (all(c(cn,"is_canceled") %in% names(df))) {
    cat(paste0("\nCancellation by ", cn, ":\n"))
    tb <- table(df[[cn]], df$is_canceled)
    print(tb)
    print( prop.table(tb, margin = 1) )
  }
}

## Booking volume by year/month
if (all(c("arrival_date_year","arrival_date_month") %in% names(df))) {
  cat("\nBookings by year & month (counts):\n")
  print( xtabs(~ arrival_date_year + arrival_date_month, data = df) )
}

## Top countries by bookings
if ("country" %in% names(df)) {
  cat("\nTop 10 countries by bookings:\n")
  cnt <- sort(table(df$country), decreasing = TRUE)
  print( head(cnt, 10) )
}

## Key metrics by hotel type
safe_mean <- function(x) mean(x, na.rm = TRUE)
safe_median <- function(x) median(x, na.rm = TRUE)

if ("hotel" %in% names(df)) {
  if ("lead_time" %in% names(df)) {
    cat("\nLead time (mean) by hotel:\n")
    print( tapply(df$lead_time, df$hotel, safe_mean) )
  }
  if ("adr" %in% names(df)) {
    cat("\nADR (mean/median) by hotel:\n")
    print( tapply(df$adr, df$hotel, safe_mean) )
    print( tapply(df$adr, df$hotel, safe_median) )
  }
  if ("stays_total_nights" %in% names(df)) {
    cat("\nTotal nights (mean) by hotel:\n")
    print( tapply(df$stays_total_nights, df$hotel, safe_mean) )
  }
}

## Lead time bins vs Cancellation
if ("lead_time" %in% names(df) && "is_canceled" %in% names(df)) {
  lt_bins <- cut(df$lead_time, breaks = c(0,7,14,30,60,90,180, Inf),
                 right = TRUE, include.lowest = TRUE)
  cat("\n[Lead time bins × Cancellation] counts:\n")
  tb_lt <- table(lt_bins, df$is_canceled); print(tb_lt)
  cat("\n[Lead time bins × Cancellation] row %:\n")
  print(prop.table(tb_lt, 1))
}

## ADR statistics
if ("adr" %in% names(df)) {
  cat("\n[ADR] overall summary:\n"); print(summary(df$adr))
  if ("hotel" %in% names(df)) {
    cat("\n[ADR] mean by hotel:\n"); print(tapply(df$adr, df$hotel, safe_mean))
    cat("\n[ADR] median by hotel:\n"); print(tapply(df$adr, df$hotel, safe_median))
  }
  if (all(c("arrival_date_year","arrival_date_month") %in% names(df))) {
    cat("\n[ADR] mean by Year × Month:\n")
    print(aggregate(adr ~ arrival_date_year + arrival_date_month, data = df, FUN = safe_mean))
  }
}

## Room type analysis
cat("\n[Levels count] reserved vs assigned room types:\n")
print(length(levels(df$reserved_room_type)))
print(length(levels(df$assigned_room_type)))

cat("\n[Top counts] reserved_room_type:\n")
print(sort(table(df$reserved_room_type), decreasing = TRUE))
cat("\n[Top counts] assigned_room_type:\n")
print(sort(table(df$assigned_room_type), decreasing = TRUE))

cat("\n[Assigned-only levels (never reserved)]:\n")
ass_only <- setdiff(levels(df$assigned_room_type), levels(df$reserved_room_type))
print(ass_only)

## Save cleaned data
write.csv(df,"/Users/lijiarui/Desktop/NTU/Analytics Strategy/Project/hotel_bookings_clean.csv",row.names=FALSE)



# Data Preparation
cancellation_data_final <- df %>%
  mutate(
    Cancellation_Status = factor(is_canceled,
                                 levels = c(0, 1),
                                 labels = c("Not Canceled", "Canceled"))
  ) %>%
  count(Cancellation_Status) %>%
  mutate(
    Percentage = n / sum(n),
    Label = paste0(round(Percentage * 100, 1), "%")
  )

# Visualization: Pie Chart Visualization: Pie Chart
ggplot(cancellation_data_final, aes(x = "", y = Percentage, fill = Cancellation_Status)) +
  geom_bar(stat = "identity", width = 1, color = "white", linewidth = 0.8) +
  coord_polar("y", start = 0) +
  geom_text(aes(label = Label),
            position = position_stack(vjust = 0.5), 
            color = "white", 
            size = 10, 
            fontface = "bold") +
  scale_fill_manual(values = c("Not Canceled" = "#5cb85c", "Canceled" = "#f0ad4e")) +
  
  labs(
    title = "Overall Hotel Reservation Cancellation Rate Distribution",
    subtitle = paste0("Total Reservations: ", comma(sum(cancellation_data_final$n))),
    fill = "Reservation Status",
    caption = "Data Source: Hotel Bookings Dataset"
  ) +
  
  theme_void(base_size = 14) + 
  
  theme(
    plot.title = element_text(face = "bold", size = 25, hjust = 0.5), # 标题增大
    plot.subtitle = element_text(size = 19, hjust = 0.5, margin = margin(b = 10)), # 副标题增大
    legend.title = element_text(face = "bold",size = 19),
    legend.text = element_text(size = 15),
    legend.position = "right"
  )


############### Analysis 2: Hotel Type and Cancellation Rate
# Data Preparation: Calculate cancellation rates by hotel type
hotel_cancellation_data <- df %>%
  mutate(
    Cancellation_Status = factor(is_canceled,
                                 levels = c(0, 1),
                                 labels = c("Not Canceled", "Canceled"))
  ) %>%
  
  group_by(hotel, Cancellation_Status) %>%
  summarise(n = n(), .groups = 'drop') %>%
  
  group_by(hotel) %>%
  mutate(
    Percentage = n / sum(n),
    
    Label = paste0(round(Percentage * 100, 1), "%")
  ) %>%
  ungroup()

# Visualization: Grouped Percentage Stacked Bar Chart
ggplot(hotel_cancellation_data, 
       aes(x = hotel, y = Percentage, fill = Cancellation_Status)) +
  geom_bar(stat = "identity", width = 0.6, alpha = 0.9, color = "white") +
  geom_text(aes(label = Label),
            position = position_stack(vjust = 0.5), 
            color = "white",
            size = 5.5,
            fontface = "bold") +
  
  scale_fill_manual(
    values = c("Canceled" = "#f0ad4e", "Not Canceled" = "#5cb85c"),
    
    breaks = c("Canceled", "Not Canceled")
  ) +
  
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  
  labs(
    title = "Cancellation Rate by Hotel Type",
    x = "Hotel Type",
    y = "Percentage of Reservations",
    fill = "Reservation Status",
    caption = "City Hotel appears to have a higher cancellation rate."
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.title.x = element_text(face = "bold", margin = margin(t = 10)),
    axis.title.y = element_text(face = "bold", margin = margin(r = 10)),
    legend.title = element_text(face = "bold", size = 18),
    legend.text = element_text(size = 14)
  )
############## 3. Advance Booking Period and Cancellation Rate

# Data Preparation: Partition into buckets and calculate the cancellation rate for each bucket.
lead_time_data <- df %>%
  # Create a lead_time bucket
  mutate(
    Lead_Time_Bins = cut(lead_time,
                         breaks = c(0, 30, 90, 180, 365, 730, Inf),
                         labels = c("0-30 Days", "31-90 Days", "91-180 Days", "181-365 Days", "1-2 Years", "> 2 Years"),
                         right = TRUE)
  ) %>%
  group_by(hotel, Lead_Time_Bins) %>%
  summarise(
    Total_Bookings = n(),
    Canceled_Count = sum(is_canceled),
    Cancellation_Rate = Canceled_Count / Total_Bookings,
    .groups = 'drop'
  ) %>%
  
  filter(Total_Bookings > 50) 


# Visualization: Bin Comparison Line Chart Bucket Comparison Line Chart
ggplot(lead_time_data, aes(x = Lead_Time_Bins, y = Cancellation_Rate, group = hotel, color = hotel)) +
  geom_line(linewidth = 1.2, alpha = 0.8) +
  geom_point(size = 3) +
  geom_text(aes(label = percent(Cancellation_Rate, accuracy = 1)),
            vjust = -1.5, 
            size = 4,
            show.legend = FALSE) + 
  
  # Add a horizontal dashed line to indicate the overall average cancellation rate.
  geom_hline(yintercept = mean(df$is_canceled), 
             linetype = "dashed", 
             color = "gray30", 
             alpha = 0.6,
             show.legend = FALSE) + 
  
  scale_color_manual(values = c("City Hotel" = "#f0ad4e", "Resort Hotel" = "#5cb85c")) +
  
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  
  labs(
    title = "Cancellation Rate by Lead Time and Hotel Type",
    x = "Lead Time Bins (Days)",
    y = "Cancellation Rate",
    color = "Hotel Type",
    caption = paste0("Dashed line represents overall mean cancellation rate (", 
                     percent(mean(df$is_canceled), accuracy = 0.1), ").")
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
    axis.title.y = element_text(face = "bold", margin = margin(r = 15)),
    legend.title = element_text(face = "bold", size = 12),
    legend.position = "top"
  )

#########4: Market Segmentation and Cancellation Rate

# Data Preparation: Calculate Churn Rate by Market Segment
market_segment_data <- df %>%
  group_by(market_segment) %>%
  summarise(
    Total_Bookings = n(),
    Canceled_Count = sum(is_canceled),
    Cancellation_Rate = Canceled_Count / Total_Bookings,
    .groups = 'drop'
  ) %>%
  filter(Total_Bookings > 100) %>%
  arrange(desc(Cancellation_Rate)) %>%
  mutate(market_segment = factor(market_segment, levels = market_segment))


# Visualization: Horizontal Lollipop Chart
ggplot(market_segment_data, aes(x = market_segment, y = Cancellation_Rate)) +
  
  geom_segment(aes(x = market_segment, xend = market_segment, y = 0, yend = Cancellation_Rate), 
               color = "gray50", 
               linewidth = 1) +
  
  geom_point(aes(color = Cancellation_Rate), size = 10) + 
  
  geom_text(aes(label = percent(Cancellation_Rate, accuracy = 1)),
            color = "white",
            size = 3.5, 
            fontface = "bold") +
  
  scale_color_gradient(low = "#f0ad4e", high = "#d9534f", labels = percent_format(accuracy = 1)) +
  
  scale_y_continuous(labels = percent_format(accuracy = 1), 
                     expand = expansion(mult = c(0, 0.1))) + 
  
  coord_flip() +
  
  labs(
    title = "Cancellation Rate by Market Segment",
    x = "Market Segment",
    y = "Cancellation Rate",
    color = "Cancellation Rate"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
    axis.title.y = element_blank(),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 15),
    legend.text = element_text(size = 12),
    panel.grid.major.y = element_blank(),
    plot.margin = margin(t = 10, r = 20, b = 10, l = 10, unit = "pt")
  )
######### 5. ADR (Average Daily Rate) and Cancellation Rate

# Data Preparation: Filtering and Preparing ADR Data
adr_data <- df %>%
  # Exclude records where ADR is 0
  filter(adr > 0) %>%
  mutate(
    Cancellation_Status = factor(is_canceled,
                                 levels = c(0, 1),
                                 labels = c("Not Canceled", "Canceled"))
  )


# Visualization: Grouped Violin Plot
ggplot(adr_data, aes(x = Cancellation_Status, y = adr, fill = Cancellation_Status)) +
  
  geom_violin(trim = TRUE, alpha = 0.7) +
  
  geom_boxplot(width = 0.15, fill = "white", color = "gray30", alpha = 0.8) +
  
  scale_fill_manual(values = c("Not Canceled" = "#5cb85c", "Canceled" = "#f0ad4e")) +
  
  coord_cartesian(ylim = c(0, 300)) + 
  
  labs(
    title = "ADR Distribution by Reservation Status",
    x = "Reservation Status",
    y = "Average Daily Rate (ADR)",
    fill = "Status"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.title.x = element_blank(), 
    axis.title.y = element_text(face = "bold", margin = margin(r = 15)),
    legend.position = "none"
  )

#######  6: Guest Count and Cancellation Rate

# Data Preparation: Calculate the total number of guests and the cancellation rate.
guests_cancellation_data <- df %>%
  mutate(total_guests = adults + children + babies) %>%
  filter(total_guests > 0 & total_guests <= 5) %>%
  
  group_by(hotel, total_guests) %>%
  summarise(
    Total_Bookings = n(),
    Canceled_Count = sum(is_canceled),
    Cancellation_Rate = Canceled_Count / Total_Bookings,
    .groups = 'drop'
  ) %>%
  mutate(total_guests = factor(total_guests))


# Visualization: Faceted Grouped Bar Chart
ggplot(guests_cancellation_data, 
       aes(x = total_guests, y = Cancellation_Rate, fill = hotel)) +
  
  geom_col(position = position_dodge(width = 0.8), 
           width = 0.7, 
           alpha = 0.8) +
  
  geom_text(aes(label = percent(Cancellation_Rate, accuracy = 1), 
                group = hotel),
            position = position_dodge(width = 0.8),
            vjust = -0.5, 
            size = 4,
            color = "gray20",
            fontface = "bold") +
  
  scale_fill_manual(values = c("City Hotel" = "#f0ad4e", "Resort Hotel" = "#5cb85c")) +
  
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  
  labs(
    title = "Cancellation Rate by Number of Guests and Hotel Type",
    x = "Total Number of Guests",
    y = "Cancellation Rate",
    fill = "Hotel Type"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
    axis.title.y = element_text(face = "bold", margin = margin(r = 15)),
    legend.title = element_text(face = "bold", size = 12),
    legend.position = "top",
    panel.grid.major.x = element_blank()
  )

######## 7: Deposit Types and Cancellation Rates

# Data Preparation: Calculate cancellation rates by deposit type
deposit_cancellation_data <- df %>%
  
  group_by(deposit_type) %>%
  summarise(
    Total_Bookings = n(),
    Canceled_Count = sum(is_canceled),
    Cancellation_Rate = Canceled_Count / Total_Bookings,
    .groups = 'drop'
  ) %>%
  
  filter(Total_Bookings > 50) %>%
  # Sort in descending order by cancellation rate
  arrange(desc(Cancellation_Rate)) %>%
  mutate(deposit_type = factor(deposit_type, levels = deposit_type))


# Visualization: Sorted Bar Chart
ggplot(deposit_cancellation_data, aes(x = deposit_type, y = Cancellation_Rate, fill = Cancellation_Rate)) +
  
  geom_col(width = 0.6, alpha = 0.9) +
  
  geom_text(aes(label = percent(Cancellation_Rate, accuracy = 1)),
            vjust = -0.5, 
            size = 5,
            color = "gray20",
            fontface = "bold") +
  
  # Set color gradient: The higher the cancellation rate, the darker the color.  
  scale_fill_gradient(low = "#f0ad4e", high = "#d9534f", guide = "none") + 
  
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  
  labs(
    title = "Cancellation Rate by Deposit Type",
    x = "Deposit Type",
    y = "Cancellation Rate"
  ) +
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(face = "bold", size = 20, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5, margin = margin(b = 10)),
    axis.title.x = element_text(face = "bold", margin = margin(t = 15)),
    axis.title.y = element_text(face = "bold", margin = margin(r = 15)),
    panel.grid.major.x = element_blank() 
  )


######## Is there any room type that was assigned but never booked?
# Identify the unique room type
unique_reserved <- unique(df$reserved_room_type)
unique_assigned <- unique(df$assigned_room_type)

# Draw a cross-tabulation heat map
room_type_crosstab <- df %>%
  # Filter out the records of actual room allocation changes that have occurred
  filter(reserved_room_type != assigned_room_type) %>%
  count(reserved_room_type, assigned_room_type) %>%
  mutate(proportion = n / sum(n))

ggplot(room_type_crosstab, aes(x = reserved_room_type, y = assigned_room_type, fill = n)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#e0f2f1", high = "#004d40", name = "Count") +
  labs(
    title = "Reservation vs. Assigned Room Type",
    x = "Reserved Room Type",
    y = "Assigned Room Type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

######## Analysis of Room Change Rate and Relationship with Repeat Customers
# Create room change marker
room_change_data <- df %>%
  mutate(
    room_changed = ifelse(assigned_room_type != reserved_room_type, "Changed", "Same"),
    is_repeated = factor(is_repeated_guest, labels = c("New Guest", "Repeated Guest"))
  ) %>%
  # Filter out the very few NA or non-standard records
  filter(!is.na(room_changed))

# Calculate the percentage of groups
room_change_summary <- room_change_data %>%
  group_by(is_repeated, room_changed) %>%
  summarise(n = n(), .groups = 'drop') %>%
  group_by(is_repeated) %>%
  mutate(
    percentage = n / sum(n),
    label = percent(percentage, accuracy = 0.1)
  ) %>%
  ungroup()

# Visualization: Group Percentage Bar Chart
ggplot(room_change_summary, aes(x = is_repeated, y = percentage, fill = room_changed)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = label), 
            position = position_stack(vjust = 0.5, reverse = TRUE),
            color = "white",
            fontface = "bold",
            size = 4) +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("Same" = "#5cb85c", "Changed" = "#f0ad4e"), name = "Room Status") +
  labs(
    title = "Room Change Rate by Guest Type",
    x = "Guest Type",
    y = "Proportion of Bookings"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))




## ===================================
## PART 1: BOOKING CANCELLATION PREDICTION
## ===================================

cat("\n========== BOOKING CANCELLATION PREDICTION ==========\n")

## ---------------------------
## TRAIN-TEST SPLIT
## ---------------------------

set.seed(0)
n <- nrow(df)
train_index <- sample(1:n, size = 0.6 * n)
train <- df[train_index, ]
test  <- df[-train_index, ]

cat("\nTraining set size:", nrow(train), "\n")
cat("Test set size:", nrow(test), "\n")
cat("Value loss rate in training:", mean(train$value_loss == "1"), "\n")
cat("Value loss rate in test:", mean(test$value_loss == "1"), "\n")

## ---------------------------
## FEATURE PREPARATION
## ---------------------------

## Remove leakage variables
drop_vars <- c("deposit_type", "reservation_status", "is_canceled")
train_clean <- train[, !(names(train) %in% drop_vars)]
test_clean  <- test[,  !(names(test) %in% drop_vars)]

## Drop constant columns
train_clean <- train_clean[, sapply(train_clean, function(x) length(unique(x)) > 1)]
test_clean  <- test_clean[, names(test_clean) %in% names(train_clean)]

## Reduce assigned_room_type cardinality
top_levels <- names(sort(table(train_clean$assigned_room_type), decreasing = TRUE))[1:5]
train_clean$assigned_room_type <- as.character(train_clean$assigned_room_type)
train_clean$assigned_room_type[!(train_clean$assigned_room_type %in% top_levels)] <- "Other"
train_clean$assigned_room_type <- factor(train_clean$assigned_room_type)

test_clean$assigned_room_type <- as.character(test_clean$assigned_room_type)
test_clean$assigned_room_type[!(test_clean$assigned_room_type %in% top_levels)] <- "Other"
test_clean$assigned_room_type <- factor(test_clean$assigned_room_type, 
                                        levels = levels(train_clean$assigned_room_type))

cat("\nFeatures being used for modeling:\n")
print(names(train_clean))

## -----------------------------------
## LOGISTIC REGRESSION MODEL
## -----------------------------------

cat("\n========== LOGISTIC REGRESSION MODEL ==========\n")

log_model <- glm(value_loss ~ ., data = train_clean, family = binomial)
summary(log_model)

## ---------------------------
## LOGISTIC REGRESSION PREDICTIONS & EVALUATION
## ---------------------------

log_pred_prob <- predict(log_model, newdata = test_clean, type = "response")
log_pred <- ifelse(log_pred_prob > 0.5, 1, 0)

## Confusion Matrix
cat("\nLogistic Regression Confusion Matrix:\n")
log_conf <- confusionMatrix(
  factor(log_pred, levels = c(0,1)),
  factor(as.numeric(as.character(test_clean$value_loss)), levels = c(0,1)),
  positive = "1"
)
print(log_conf)

## ROC & AUC
true_labels <- as.numeric(test_clean$value_loss) - 1
roc_log <- roc(true_labels, log_pred_prob)
auc_log <- auc(roc_log)
cat("\nLogistic Regression AUC:", auc_log, "\n")

## PR Curve
fg_log <- log_pred_prob[true_labels == 1]
bg_log <- log_pred_prob[true_labels == 0]
pr_log <- pr.curve(scores.class0 = fg_log, scores.class1 = bg_log, curve = TRUE)
cat("Logistic Regression PR-AUC:", pr_log$auc.integral, "\n")

## -----------------------------------
## RANDOM FOREST MODEL
## -----------------------------------

cat("\n========== RANDOM FOREST MODEL ==========\n")

## Handle class imbalance
class_weights <- c("0" = 1, "1" = 3)

set.seed(140522)
rf_model <- randomForest(
  value_loss ~ ., 
  data = train_clean, 
  ntree = 100,
  mtry = sqrt(ncol(train_clean) - 1),
  importance = TRUE,
  classwt = class_weights
)

print(rf_model)

## ---------------------------
## RANDOM FOREST PREDICTIONS & EVALUATION
## ---------------------------

rf_pred_prob <- predict(rf_model, newdata = test_clean, type = "prob")[,2]
rf_pred <- ifelse(rf_pred_prob > 0.5, 1, 0)

## Confusion Matrix
cat("\nRandom Forest Confusion Matrix:\n")
rf_conf <- confusionMatrix(
  factor(rf_pred, levels = c(0,1)),
  factor(as.numeric(as.character(test_clean$value_loss)), levels = c(0,1)),
  positive = "1"
)
print(rf_conf)

## ROC & AUC
roc_rf <- roc(true_labels, rf_pred_prob)
auc_rf <- auc(roc_rf)
cat("\nRandom Forest AUC:", auc_rf, "\n")

## PR Curve
fg_rf <- rf_pred_prob[true_labels == 1]
bg_rf <- rf_pred_prob[true_labels == 0]
pr_rf <- pr.curve(scores.class0 = fg_rf, scores.class1 = bg_rf, curve = TRUE)
cat("Random Forest PR-AUC:", pr_rf$auc.integral, "\n")

## -----------------------------------
## XGBOOST MODEL
## -----------------------------------

cat("\n========== XGBOOST MODEL ==========\n")

## One-hot encode categorical features
train_matrix <- model.matrix(value_loss ~ . -1, data = train_clean)
test_matrix  <- model.matrix(value_loss ~ . -1, data = test_clean)

train_label <- as.numeric(as.character(train_clean$value_loss))
test_label  <- as.numeric(as.character(test_clean$value_loss))

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix,  label = test_label)

## Handle class imbalance with scale_pos_weight
tbl <- table(train_label)
scale_pos_weight <- as.numeric(tbl["0"] / tbl["1"])

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.05,
  scale_pos_weight = scale_pos_weight
)

set.seed(42)
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 20,
  verbose = 1,
  print_every_n = 50
)

## ---------------------------
## XGBOOST PREDICTIONS & EVALUATION
## ---------------------------

xgb_pred_prob <- predict(bst, dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)

## Confusion Matrix
cat("\nXGBoost Confusion Matrix:\n")
xgb_conf <- confusionMatrix(
  factor(xgb_pred, levels = c(0,1)),
  factor(test_label, levels = c(0,1)),
  positive = "1"
)
print(xgb_conf)

## ROC & AUC
roc_xgb <- roc(test_label, xgb_pred_prob)
auc_xgb <- auc(roc_xgb)
cat("\nXGBoost AUC:", auc_xgb, "\n")

## PR Curve
fg_xgb <- xgb_pred_prob[test_label == 1]
bg_xgb <- xgb_pred_prob[test_label == 0]
pr_xgb <- pr.curve(scores.class0 = fg_xgb, scores.class1 = bg_xgb, curve = TRUE)
cat("XGBoost PR-AUC:", pr_xgb$auc.integral, "\n")

## --------------------------------
## MODEL COMPARISON - BOOKING CANCELLATION
## --------------------------------

cat("\n========== BOOKING CANCELLATION MODEL COMPARISON ==========\n")

## Compile results
cancellation_results <- data.frame(
  Model     = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy  = c(log_conf$overall["Accuracy"], 
                rf_conf$overall["Accuracy"], 
                xgb_conf$overall["Accuracy"]),
  AUC       = c(auc_log, auc_rf, auc_xgb),
  PR_AUC    = c(pr_log$auc.integral, pr_rf$auc.integral, pr_xgb$auc.integral),
  Precision = c(log_conf$byClass["Precision"], 
                rf_conf$byClass["Precision"], 
                xgb_conf$byClass["Precision"]),
  Recall    = c(log_conf$byClass["Recall"], 
                rf_conf$byClass["Recall"], 
                xgb_conf$byClass["Recall"]),
  F1        = c(log_conf$byClass["F1"], 
                rf_conf$byClass["F1"], 
                xgb_conf$byClass["F1"])
)

print(cancellation_results)

## ---------------------------
## VISUALIZATION - ROC CURVES
## ---------------------------

par(mfrow = c(2, 2))

## Logistic Regression ROC
plot(roc_log, main = paste("Logistic Regression ROC (AUC:", round(auc_log, 3), ")"), 
     col = "darkred", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

## Random Forest ROC
plot(roc_rf, main = paste("Random Forest ROC (AUC:", round(auc_rf, 3), ")"), 
     col = "darkgreen", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

## XGBoost ROC
plot(roc_xgb, main = paste("XGBoost ROC (AUC:", round(auc_xgb, 3), ")"), 
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

## Combined ROC
plot(roc_log, col = "darkred", lwd = 2, main = "Combined ROC Curves")
lines(roc_rf, col = "darkgreen", lwd = 2)
lines(roc_xgb, col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright", 
       legend = c("Logistic Reg", "Random Forest", "XGBoost"),
       col = c("darkred", "darkgreen", "blue"), lwd = 2)

## ------------------------------
## FEATURE IMPORTANCE ANALYSIS - BOOKING CANCELLATION
## ------------------------------

cat("\n========== FEATURE IMPORTANCE - BOOKING CANCELLATION ==========\n")

## Random Forest Feature Importance
cat("\nRandom Forest Feature Importance:\n")
rf_imp <- importance(rf_model)
rf_imp_df <- data.frame(
  Feature = rownames(rf_imp),
  MeanDecreaseAccuracy = rf_imp[,3],
  MeanDecreaseGini = rf_imp[,4]
)
rf_imp_df <- rf_imp_df[order(-rf_imp_df$MeanDecreaseGini), ]
print(head(rf_imp_df, 15))

## XGBoost Feature Importance
cat("\nXGBoost Feature Importance:\n")
xgb_imp <- xgb.importance(model = bst)
print(head(xgb_imp, 15))

## Logistic Regression Coefficients
cat("\nLogistic Regression Significant Coefficients:\n")
log_coef <- summary(log_model)$coefficients
log_df <- data.frame(
  Feature = rownames(log_coef)[-1],
  Estimate = log_coef[-1, "Estimate"],
  p_value = log_coef[-1, "Pr(>|z|)"]
)

## Keep significant features
meaningful_features <- c(
  "hotelResort Hotel", "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
  "children", "is_repeated_guest1", "previous_cancellations",
  "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
  "required_car_parking_spaces", "mealSC", "total_of_special_requests", 
  "customer_typeTransient", "distribution_channelTA/TO",
  "market_segmentOnline TA", "adults"
)
log_top <- subset(log_df, Feature %in% meaningful_features & p_value < 0.05)
print(log_top)

## ---------------------------
## VISUALIZATION - FEATURE IMPORTANCE
## ---------------------------

## Random Forest
rf_imp_top <- rf_imp_df[1:10, ]
p1 <- ggplot(rf_imp_top, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance", 
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()

## XGBoost
xgb_imp_top <- xgb_imp[1:10, ]
p2 <- ggplot(xgb_imp_top, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "XGBoost Feature Importance", 
       x = "Feature", y = "Gain") +
  theme_minimal()

## Logistic Regression
log_top$Direction <- ifelse(log_top$Estimate > 0, "Positive", "Negative")
p3 <- ggplot(log_top, aes(x = reorder(Feature, Estimate), y = Estimate, fill = Direction)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("Negative" = "red", "Positive" = "darkgreen")) +
  labs(title = "Logistic Regression Coefficients",
       x = "Feature", y = "Coefficient") +
  theme_minimal()

grid.arrange(p1, p2, p3, ncol = 2)

## ===================================
## PART 2: REPEAT GUEST PREDICTION 
## ===================================

cat("\n========== REPEAT GUEST PREDICTION ==========\n")

## Reset dataset to cleaned version
df <- df_final_cleaned

df$is_repeated_guest <- as.factor(df$is_repeated_guest)
df$distribution_channel <- as.factor(df$distribution_channel)
df$market_segment <- as.factor(df$market_segment)

cat("Distribution Channels:\n")
print(table(df$distribution_channel))
cat("\nMarket Segments:\n")
print(table(df$market_segment))

## ---------------------------
## FEATURE PREPARATION
## ---------------------------

## Drop leakage / post-booking / collinear columns
columns_to_drop <- c("arrival_date", "arrival_date_year", "arrival_date_month",
                     "arrival_date_week_number", "arrival_date_day_of_month",
                     "reservation_status", "is_canceled", "total_guests", 
                     "stays_total_nights", "value_loss")

columns_to_drop <- columns_to_drop[columns_to_drop %in% names(df)]
df <- df[, !names(df) %in% columns_to_drop]

cat("\nFeatures being used for modeling:\n")
print(names(df))
cat("\nConfirming distribution_channel is included:", "distribution_channel" %in% names(df), "\n")
cat("Confirming market_segment is included:", "market_segment" %in% names(df), "\n")

## ---------------------------
## TRAIN-TEST SPLIT
## ---------------------------

set.seed(123)
train_index <- sample(1:nrow(df), size = 0.7 * nrow(df))
train <- df[train_index, ]
test  <- df[-train_index, ]

cat("\nTraining set size:", nrow(train), "\n")
cat("Test set size:", nrow(test), "\n")
cat("Repeat guest rate in training:", mean(train$is_repeated_guest == "1"), "\n")
cat("Repeat guest rate in test:", mean(test$is_repeated_guest == "1"), "\n")

## -----------------------------------
## RANDOM FOREST MODEL
## -----------------------------------

cat("\n========== RANDOM FOREST MODEL ==========\n")

## Set class weights to handle imbalance
class_weights <- c("0" = 1, "1" = 20)

set.seed(42)
rf_repeat_model <- randomForest(
  is_repeated_guest ~ ., 
  data = train, 
  ntree = 100,
  mtry = sqrt(ncol(train) - 1),
  importance = TRUE,
  classwt = class_weights,
  nodesize = 5
)

print(rf_repeat_model)

## ---------------------------
## RANDOM FOREST PREDICTIONS & EVALUATION
## ---------------------------

rf_repeat_pred <- predict(rf_repeat_model, newdata = test)
rf_repeat_prob <- predict(rf_repeat_model, newdata = test, type = "prob")[,2]

## Confusion Matrix on full test set
cat("\nRandom Forest Confusion Matrix (Full Test Set):\n")
rf_repeat_conf <- confusionMatrix(rf_repeat_pred, test$is_repeated_guest, positive = "1")
print(rf_repeat_conf)

## ROC & AUC
roc_rf_repeat <- roc(as.numeric(as.character(test$is_repeated_guest)), rf_repeat_prob)
auc_rf_repeat <- auc(roc_rf_repeat)
cat("\nRandom Forest AUC:", auc_rf_repeat, "\n")

## PR Curve
fg_rf_repeat <- rf_repeat_prob[test$is_repeated_guest == "1"]
bg_rf_repeat <- rf_repeat_prob[test$is_repeated_guest == "0"]
pr_rf_repeat <- pr.curve(scores.class0 = fg_rf_repeat, scores.class1 = bg_rf_repeat, curve = TRUE)
cat("Random Forest PR-AUC:", pr_rf_repeat$auc.integral, "\n")

## Evaluation on balanced test set
test_min <- test[test$is_repeated_guest == "1", ]
test_maj <- test[test$is_repeated_guest == "0", ]
set.seed(42)
test_maj_sample <- test_maj[sample(1:nrow(test_maj), nrow(test_min)), ]
test_balanced <- rbind(test_min, test_maj_sample)

rf_repeat_pred_bal <- predict(rf_repeat_model, newdata = test_balanced)
cat("\nRandom Forest Confusion Matrix (Balanced Test Set):\n")
rf_repeat_conf_bal <- confusionMatrix(rf_repeat_pred_bal, test_balanced$is_repeated_guest, positive = "1")
print(rf_repeat_conf_bal)

## -----------------------------------
## XGBOOST MODEL WITH BALANCED DATASET
## -----------------------------------

cat("\n========== XGBOOST MODEL ==========\n")

## Prepare features for XGBoost
df_xgb <- df

## One-hot encode categorical variables
cat("\nOne-hot encoding categorical variables for XGBoost...\n")

target <- df_xgb$is_repeated_guest
df_features <- df_xgb %>% select(-is_repeated_guest)

cat("Categorical variables to be encoded:\n")
cat_vars <- names(df_features)[sapply(df_features, is.factor)]
print(cat_vars)

## One-hot encode
dmy <- dummyVars(" ~ .", data = df_features)
df_mat <- data.frame(predict(dmy, newdata = df_features))

cat("\nFeatures after one-hot encoding:", ncol(df_mat), "\n")

## Check for distribution_channel and market_segment encoded features
dist_features <- names(df_mat)[grep("distribution_channel", names(df_mat))]
market_features <- names(df_mat)[grep("market_segment", names(df_mat))]
cat("\nDistribution channel encoded features:", length(dist_features), "\n")
cat("Market segment encoded features:", length(market_features), "\n")

## ---------------------------
## BALANCE DATASET FOR XGBOOST
## ---------------------------

## Combine features with target
df_combined <- cbind(df_mat, target = as.numeric(as.character(target)))

## Split by class
class0 <- df_combined[df_combined$target == 0, ]
class1 <- df_combined[df_combined$target == 1, ]

cat("\nOriginal class distribution:\n")
cat("Class 0 (non-repeat):", nrow(class0), "\n")
cat("Class 1 (repeat):", nrow(class1), "\n")

## Downsample majority class
set.seed(42)
class0_sample <- class0[sample(1:nrow(class0), nrow(class1)), ]
df_balanced <- rbind(class0_sample, class1)

cat("\nBalanced dataset:\n")
cat("Class 0 (non-repeat):", sum(df_balanced$target == 0), "\n")
cat("Class 1 (repeat):", sum(df_balanced$target == 1), "\n")

## ---------------------------
## XGBOOST TRAINING
## ---------------------------

## Train-test split on balanced data
set.seed(42)
train_index_xgb <- sample(1:nrow(df_balanced), size = 0.7 * nrow(df_balanced))
train_xgb <- df_balanced[train_index_xgb, ]
test_xgb  <- df_balanced[-train_index_xgb, ]

## Prepare matrices
train_matrix_xgb <- as.matrix(train_xgb[, -which(names(train_xgb) == "target")])
test_matrix_xgb  <- as.matrix(test_xgb[, -which(names(test_xgb) == "target")])

train_label_xgb <- train_xgb$target
test_label_xgb  <- test_xgb$target

dtrain_xgb <- xgb.DMatrix(data = train_matrix_xgb, label = train_label_xgb)
dtest_xgb  <- xgb.DMatrix(data = test_matrix_xgb,  label = test_label_xgb)

## Set parameters
params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.05
)

set.seed(42)
bst_repeat <- xgb.train(
  params = params_xgb,
  data = dtrain_xgb,
  nrounds = 200,
  watchlist = list(train = dtrain_xgb, test = dtest_xgb),
  early_stopping_rounds = 20,
  verbose = 1,
  print_every_n = 50
)

## ---------------------------
## XGBOOST PREDICTIONS & EVALUATION
## ---------------------------

xgb_repeat_prob <- predict(bst_repeat, dtest_xgb)
xgb_repeat_pred <- ifelse(xgb_repeat_prob > 0.5, 1, 0)

## Confusion Matrix
cat("\nXGBoost Confusion Matrix (Balanced Test Set):\n")
xgb_repeat_conf <- confusionMatrix(
  factor(xgb_repeat_pred, levels = c(0,1)),
  factor(test_label_xgb, levels = c(0,1)),
  positive = "1"
)
print(xgb_repeat_conf)

## ROC & AUC
roc_xgb_repeat <- roc(test_label_xgb, xgb_repeat_prob)
auc_xgb_repeat <- auc(roc_xgb_repeat)
cat("\nXGBoost AUC:", auc_xgb_repeat, "\n")

## PR Curve
fg_xgb_repeat <- xgb_repeat_prob[test_label_xgb == 1]
bg_xgb_repeat <- xgb_repeat_prob[test_label_xgb == 0]
pr_xgb_repeat <- pr.curve(scores.class0 = fg_xgb_repeat, scores.class1 = bg_xgb_repeat, curve = TRUE)
cat("XGBoost PR-AUC:", pr_xgb_repeat$auc.integral, "\n")

## --------------------------------
## MODEL COMPARISON - REPEAT GUEST
## --------------------------------

cat("\n========== REPEAT GUEST MODEL COMPARISON ==========\n")

repeat_results <- data.frame(
  Model = c("Random Forest", "XGBoost"),
  Accuracy = c(rf_repeat_conf_bal$overall["Accuracy"], xgb_repeat_conf$overall["Accuracy"]),
  AUC = c(auc_rf_repeat, auc_xgb_repeat),
  PR_AUC = c(pr_rf_repeat$auc.integral, pr_xgb_repeat$auc.integral),
  Precision = c(rf_repeat_conf_bal$byClass["Precision"], xgb_repeat_conf$byClass["Precision"]),
  Recall = c(rf_repeat_conf_bal$byClass["Recall"], xgb_repeat_conf$byClass["Recall"]),
  F1 = c(rf_repeat_conf_bal$byClass["F1"], xgb_repeat_conf$byClass["F1"])
)

print(repeat_results)

## ---------------------------
## VISUALIZATION - REPEAT GUEST ROC CURVES
## ---------------------------

par(mfrow = c(1, 2))

## Random Forest ROC
plot(roc_rf_repeat, main = paste("Random Forest ROC (AUC:", round(auc_rf_repeat, 3), ")"), 
     col = "darkgreen", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

## XGBoost ROC
plot(roc_xgb_repeat, main = paste("XGBoost ROC (AUC:", round(auc_xgb_repeat, 3), ")"), 
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

## ------------------------------
## FEATURE IMPORTANCE ANALYSIS - REPEAT GUEST
## ------------------------------

cat("\n========== FEATURE IMPORTANCE - REPEAT GUEST ==========\n")

## Random Forest Feature Importance
cat("\nRandom Forest Feature Importance:\n")
rf_imp_repeat <- importance(rf_repeat_model)
rf_imp_repeat_df <- data.frame(
  Feature = rownames(rf_imp_repeat),
  MeanDecreaseAccuracy = rf_imp_repeat[,3],
  MeanDecreaseGini = rf_imp_repeat[,4]
)
rf_imp_repeat_df <- rf_imp_repeat_df[order(-rf_imp_repeat_df$MeanDecreaseGini), ]
print(head(rf_imp_repeat_df, 15))

## XGBoost Feature Importance
cat("\nXGBoost Feature Importance:\n")
xgb_imp_repeat <- xgb.importance(model = bst_repeat)
print(head(xgb_imp_repeat, 15))

## ---------------------------
## VISUALIZATION - REPEAT GUEST FEATURE IMPORTANCE
## ---------------------------

## Random Forest
rf_imp_repeat_top <- rf_imp_repeat_df[1:10, ]
p4 <- ggplot(rf_imp_repeat_top, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance", 
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()

## XGBoost
xgb_imp_repeat_top <- xgb_imp_repeat[1:10, ]
p5 <- ggplot(xgb_imp_repeat_top, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "XGBoost Feature Importance", 
       x = "Feature", y = "Gain") +
  theme_minimal()

grid.arrange(p4, p5, ncol = 2)

## ===================================
## FINAL SUMMARY
## ===================================

cat("\n========== FINAL SUMMARY ==========\n")

cat("\nBOOKING CANCELLATION PREDICTION RESULTS:\n")
print(cancellation_results)

cat("\nREPEAT GUEST PREDICTION RESULTS:\n")
print(repeat_results)



## ===================================
## Prediction ADR
## ===================================
df <- df%>% filter(adr<3000) # Remove outliers
df <- df %>% select(-arrival_date)
df$reserved_room_type <- as.character(df$reserved_room_type)
df$assigned_room_type <- as.character(df$assigned_room_type)
df$assigned_room_type[!df$assigned_room_type %in% c("A", "B","C", "D", "E", "F", "G")] <- "Other"
df$reserved_room_type[!df$reserved_room_type %in% c("A", "B","C", "D", "E", "F", "G")] <- "Other"
df <- dummy_columns(df[,], remove_first_dummy = T, 
                    remove_selected_columns = T)
df.scaled <- scale(df, center=T, scale=T)




# Split the train dataset and test dataset (Train: 70%, Test: 30%)
set.seed(123)
idx <- sample(1:nrow(df), nrow(df)*0.7)
train <- df[idx, ]
test <- df[-idx, ]

# Prediction Model 1: Liear Regression

# Fit regression model
m1 <- lm(adr~., train)
summary(m1)

m2 <- step(m1, direction="backward", trace=0) # Stepwise model selection: utomatic procedure that adds or removes predictors to find a simpler model with a better AIC
summary(m2)

# Check on Multicollinearity Using GVIF 
# The usual Variance Inflation Factor (VIF) isn’t directly applicable because these variables have multiple degrees of freedom (Df).
# Calculate VIF (replace m2 with your model name)
v <- vif(m2)

# Handle GVIF case (when there are categorical variables)
if (is.matrix(v)) {
  vif_values <- (v[, "GVIF"])^(1/(2*v[, "Df"]))  # Adjust GVIF
  names(vif_values) <- rownames(v)
} else {
  vif_values <- v  # Simple VIF for numeric predictors
}

# Filter high VIF variables
names(vif_values[vif_values > 5]) # arrival_date_week_number which is 35.67 (exceed the cut-off 5, which definite collinearity issue) 
m3 <- update(m2, . ~ . - arrival_date_week_number)
vif(m3) # Checked all the GVIF for all variables below cut-off 2, which eliminate the effect from collinearity

# Check on Root Mean Squared Erro
sqrt(mean(m3$residuals^2)) # train RMSE
preds <- predict(m3, test)
sqrt(mean((test$adr-preds)^2)) # test RMSE

# Prediction Model 2: Random Forest 
set.seed(42)

# Choose mtry = sqrt(p) where p excludes the target
p <- ncol(train) - 1
mtry_val <- max(1, floor(sqrt(p)))

# Step 1. Fit the full (complex) tree with 10-fold CV
set.seed(123)  # for reproducibility
m_rf <- rpart(
  adr ~ .,
  data = train,
  method = "anova",  # assuming 'adr' is numeric
  control = rpart.control(
    cp = 0.01,       # small cp to grow a large tree first
    minsplit = 5,
    xval = 10          # 10-fold cross-validation
  )
)

# Step 2. Visualize the complexity parameter (CP) plot
plotcp(m_rf)
printcp(m_rf)

# Step 3. Apply the 1-SE Rule for pruning
# Find the row with minimum xerror
min_xerror <- min(m_rf$cptable[,"xerror"])

# Calculate threshold = min error + 1 standard error
se_threshold <- min_xerror + m_rf$cptable[which.min(m_rf$cptable[,"xerror"]),"xstd"]

# Find the simplest tree (smallest complexity) within 1-SE
bestcp_1se <- m_rf$cptable[m_rf$cptable[,"xerror"] <= se_threshold, "CP"][1]

bestcp_1se

# Step 4. Prune the tree using the 1-SE rule
m_pruned <- prune(m_rf, cp = bestcp_1se)

m_pruned$variable.importance # Checking on the importance of trees 

# Step 5. Visualisation
rpart.plot(
  m_pruned,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE,
  cex = 0.6  # smaller font if the plot is still dense
)

fancyRpartPlot(m_pruned, sub = "") # Visual the trees clearly

# train RMSE
preds <- predict(m_pruned, train)
sqrt(mean((train$adr-preds)^2)) # train dataset RMSE
preds <- predict(m_pruned, test)
sqrt(mean((test$adr-preds)^2)) # test dataset RMSE


## ===================================
## Customer Segmentation (Clustering)
## ===================================
# Find best k value 
set.seed(123)
k <- 1:15
tot <- c()
for(i in k){
  model <- kmeans(df.scaled, i)
  tot[i] <- model$tot.withinss
}
plot(k, tot, type="b")
# The curve shows a clear bend at k = 4, after which the reduction in within-cluster, we choose k = 4.

# Clustering 
k4 <-  kmeans(df.scaled, 4)
aggregate(df, by=list(cluster=k4$cluster), mean) #Aggregate for means, find the "cluster centroids" in the original variable space
fviz_cluster(k4, data = df) 

# 1) Select clustering features (robustly by patterns)
# We cluster on behavioral, spending, and channel features and exclude outcomes/IDs/seasonality
# to avoid leakage and noise; rare “_undefined” dummies are removed to improve stability.
keep_patterns <- c(
  "^booking_changes$", "^previous_cancellations$", "^days_in_waiting_list$",
  "^stays_in_weekend_nights$", "^stays_in_week_nights$", "^stays_total_nights$",
  "^total_of_special_requests$", "^required_car_parking_spaces$", "^total_guests$",
  "^adults$", "^children$", "^babies$", "^is_repeated_guest$", "^adr$",
  "^meal_", "^market_segment_", "^distribution_channel_", "^customer_type_", "^deposit_type_"
)

# columns to exclude (outcomes / seasonality / IDs / noise)
drop_patterns <- c(
  "^is_canceled$", "^reservation_status_", "^arrival_date_", "^hotel_",
  "^assigned_room_type_", "^reserved_room_type_", "_undefined$", "^lead_time$"
)

keep_idx  <- Reduce(`|`, lapply(keep_patterns,  grepl, x = names(df)))
drop_idx  <- Reduce(`|`, lapply(drop_patterns,  grepl, x = names(df)))
final_cols <- names(df)[keep_idx & !drop_idx]

df_clust <- df[, final_cols, drop = FALSE]

# 2) Scale (Use Z-score to exclude outliers, Z-score > 3)
df_clust_scaled <- as.data.frame(scale(df_clust, center = TRUE, scale = TRUE))
summary(df_clust_scaled)

df_clust_scaled <- df_clust_scaled[
  df_clust_scaled$stays_in_weekend_nights <= 3 &
    df_clust_scaled$stays_in_week_nights <= 3 &
    df_clust_scaled$adults <= 3 &
    df_clust_scaled$adr <= 3 &
    df_clust_scaled$previous_cancellations <= 3 &
    df_clust_scaled$booking_changes <= 3, 
]

summary(df_clust_scaled)


# 3) K-means with k = 4
set.seed(123)
k4 <- kmeans(df_clust_scaled, centers = 4, nstart = 25)

# 5) Visualize clusters using the scaled matrix used for fitting
fviz_cluster(k4, data = df_clust_scaled,
             geom = "point", ellipse.type = "convex",
             ggtheme = theme_minimal())

# Just scatter points (no cluster boundary polygon)
fviz_cluster(k4, data = df_clust_scaled,
             geom = "point",
             ellipse = FALSE,
             show.clust.cent = TRUE)

# PCA Analysis 
pca_res <- PCA(df_clust_scaled, graph = FALSE)
round(pca_res$var$coord[, 1:2], 2)

fviz_contrib(pca_res, choice = "var", axes = 1)
fviz_contrib(pca_res, choice = "var", axes = 2)


