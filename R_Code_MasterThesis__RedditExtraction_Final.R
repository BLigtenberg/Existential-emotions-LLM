#install packages
install.packages("RedditExtractoR")
install.packages("tidyverse")
install.packages("dplyr")
install.packages("stringr")
install.packages("writexl")
install.packages("cld2")

library(RedditExtractoR)
library(tidyverse)
library(dplyr)
library(stringr)
library(writexl)
library(cld2)

#creating an array of subreddits
subreddits <- c(
  "existentialism",
  "existential_crisis",
  "ChronicIllness",
  "depression",
  "lonely",
  "meaningoflife",
  "freewill",
  "solitude",
  "SeriousConversation",
  "stoicism")

#create list in which the data of all subreddits' threads is saved in
analysis_data <- list()

#for loop to scrape and filter subreddit threads
for (sub in subreddits){
  subthread_top <- find_thread_urls(
    subreddit = sub,
    sort_by = "top", 
    period = "all")
  
  Sys.sleep(20)
  
  subthread_new <- find_thread_urls(
    subreddit = sub,
    sort_by = "new", 
    period = "all")
  
  Sys.sleep(20)
  
  top_posts <- subthread_top %>% 
    slice_head(n = 160) %>%
    mutate(source = "top")
  print(paste(sub, "- top posts:", nrow(top_posts)))
  
  new_posts <- subthread_new %>% 
    slice_head(n = 100) %>%
    mutate(source = "new")
  print(paste(sub, "- new posts:", nrow(new_posts)))
  
  sub_data_raw <- bind_rows(
    top_posts, 
    new_posts)
  
  analysis_data[[sub]] <-  sub_data_raw
}

#merge list into a dataframe
posts_2000 <- bind_rows(analysis_data)

#check numbers of unfiltered posts in dataframe
print(paste("number of unfiltered posts:", nrow(posts_2000)))

#check 80/20 ratio per subreddit
posts_2000 %>%
  count(subreddit, source) %>%
  print()

#check column-names of dataframe
names(posts_2000)

#filter duplicates based on URLs
unique_posts <- posts_2000 %>%
  distinct(url,.keep_all = TRUE)
print(
  paste(
    "number of posts left after filtering for double appearances of URLs :", 
    nrow(unique_posts)
  )
)

#remove username and URLs in text and title

unique_posts <- unique_posts %>%
  mutate(
    title = str_replace_all(title, "u/\\S+|/u/\\S+|@\\S+", " "), 
    text = str_replace_all(text, "u/\\S+|/u/\\S+|@\\S+", " "),
    title = str_replace_all(title, "https?://\\S+|www\\.\\S+", " "),
    text = str_replace_all(text, "https?://\\S+|www\\.\\S+", " ")
  )

#filter posts < 50 and merge title and text into a single column
complete_posts <- unique_posts %>%
  mutate(
    title = str_squish(coalesce(title, "")),
    text = str_squish(coalesce(text, "")),
    title_wordcount = str_count(title, "\\S+"),
    text_wordcount = str_count(text, "\\S+"),
    total_wordcount = title_wordcount + text_wordcount,
    title_text_combined = str_squish(paste(title, "[SEP]", text))
  ) %>%
  filter(total_wordcount >= 50)


#Removing username, URLs, date, and timestamp
complete_posts <- complete_posts%>%
  select(-any_of(c("author", "url", "date_utc", "timestamp")))


#filter for English only
cleaned_posts <- complete_posts %>%
  mutate(
    language = cld2::detect_language(
      title_text_combined,
      plain_text = TRUE,
      lang_code = TRUE
    ) 
  ) %>%
  filter(language == "en")

#add post_IDs
cleaned_posts <- cleaned_posts %>%
  mutate(
    post_id = sprintf("POST_%04d", row_number()))

#save filtered total dataaset as an Excel-file
df <- cleaned_posts
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Filtered_total_posts.xlsx")

#draw 50 posts per subreddit from total dataset (40 top and 10 new posts per subreddit)
#extracting 40 top posts per subreddit
top_posts_ratio <- cleaned_posts %>%
  filter(source == "top") %>%
  group_by(subreddit) %>%
  slice_head(n = 40)

#extracting 10 new posts per subreddit
new_posts_ratio <- cleaned_posts %>%  
  filter(source == "new") %>%
  group_by(subreddit) %>%
  slice_head(n = 10)

final_dataset <- bind_rows(top_posts_ratio, new_posts_ratio) %>%
  ungroup()

final_dataset %>%
  count(subreddit, source) %>%
  print()

#save filtered total dataset as an Excel-file
df <- final_dataset
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Reddit_Existential_Dataset_Final.xlsx")


#assign fairly distributed number of posts in random order to the datasets
set.seed(123)

#shuffle posts wihin each subreddit and source group
final_dataset_shuffled <- final_dataset %>%
  group_by(subreddit, source) %>%
  slice_sample(prop = 1) %>%
  ungroup()

#create test set with 4 top posts and 1 new post per subreddit
test_top <- final_dataset_shuffled %>%
  filter(source == "top") %>%
  group_by(subreddit) %>%
  slice_head(n = 4) %>%
  ungroup()

test_new <- final_dataset_shuffled %>%
  filter(source == "new") %>%
  group_by(subreddit) %>%
  slice_head(n = 1) %>%
  ungroup()

test_set <- bind_rows(test_top, test_new) %>%
  slice_sample(prop = 1)

#remove allocated test posts from the dataset
remaining_posts <- final_dataset_shuffled %>%
  anti_join(test_set, by = "post_id")


#create validation set with 4 top posts and 1 new post per subreddit
val_top <- remaining_posts %>%
  filter(source == "top") %>%
  group_by(subreddit) %>%
  slice_head(n = 4) %>%
  ungroup()

val_new <- remaining_posts %>%
  filter(source == "new") %>%
  group_by(subreddit) %>%
  slice_head(n = 1) %>%
  ungroup()

validation_set <- bind_rows(val_top, val_new) %>%
  slice_sample(prop = 1)

#remove allocated validation posts from the dataset
left_posts <- remaining_posts %>%
  anti_join(validation_set, by = "post_id")


#create training set with 32 top posts and 8 new post per subreddit

training_set <- left_posts %>%
  slice_sample(prop = 1)

  
print(paste("Number of posts in test set:", nrow(test_set)))

test_set %>%
  count(subreddit, source) %>%
  print()

df <- test_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Test_set_Raw.xlsx")

print(paste("Number of posts in validation set:", nrow(validation_set)))

validation_set %>%
  count(subreddit, source) %>%
  print()

df <- validation_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Validation_set_Raw.xlsx")

print(paste("Number of posts in training set:", nrow(training_set)))

training_set %>%
  count(subreddit, source) %>%
  print()

df <- training_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Training_set_Raw.xlsx")


#filter columns with information about subreddit, source, comments and language from the test set, validation set and training set  
cleaned_test_set <- test_set %>%
  select(-subreddit, -comments, -source, -language, -title_wordcount, -text_wordcount, -total_wordcount)

cleaned_validation_set <- validation_set %>%
  select(-subreddit, -comments, -source, -language,-title_wordcount, -text_wordcount, -total_wordcount)

cleaned_training_set <- training_set %>%
  select(-subreddit, -comments, -source, -language,-title_wordcount, -text_wordcount, -total_wordcount)


df <- cleaned_test_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Test_set_cleaned_Final.xlsx")

df <- cleaned_validation_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Validation_set_cleaned_Final.xlsx")

df <- cleaned_training_set
write_xlsx(
  df, 
  "C:/Users/rebec/OneDrive/Dokumente/Training_set_cleaned_Final.xlsx")


nrow(inner_join(test_set, validation_set, by = "post_id"))
nrow(inner_join(test_set, training_set, by = "post_id"))
nrow(inner_join(validation_set, training_set, by = "post_id"))
