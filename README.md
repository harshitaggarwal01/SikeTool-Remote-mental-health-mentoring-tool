# Octahacks-Team Coderaptors

## Sike-Tool - Remote Mental Health Monitoring Tool

### The Problem
The COVID-19 pandemic has had a number of consequences--one of which is the population's mental health. Many are experiencing poor mental health for the first time while others have lost access to some of the resources they used to manage their mental health. Namely, a plethora of individuals struggling with depression, anxiety or other mental illnesses have lost access to their regular appointments with their therapists or psychiatrists.

Unfortunately, online video appointments are not feasible for individuals with limited or poor internet access. Without analyzing the body language of their patients, therapists will not be able to accurately glean their patients' health. Our remote mental health monitoring tool aims to solve this issue.

### The Solution
Our web application aims to analyze the user's mental health using a common therapy technique--journaling. The user will submit diary entries into our application. In the therapist-patient scenario, they will be required to keep their camera on. 

#### How It Works (Tentatively):
  1. The patient grants permissions to the web app for camera access.
  
  2. When the patient starts typing, the web app will be taking a face capture after each sentence. These face captures are fed into our facial emotion recognition    model to analyze the body language of the user as they continue writing.
  
  3. The patient, after completing their journal entry, will press the "Summarize" button to submit their entry. On the back-end, three models are in action: (1) Summarizer (2) Sentiment Analyzer (3) Depression or Not Binary Classifier
  
  4. The patient will see their original entry, their summarized entry, and the likelihood they are depressed in the analyze page of our web application. If the likelihood exceeds a particular threshold value, they will be given access to links to resources that can help them manage their depression at home. If the patient has submitted entries in the past, they can return to an old entry as well. 
  
  5. On the admin side, the therapist can request a summary of a particular user's depression indication likelihood over a period of time. We will not be diagnosing the patient but rather provide a recommendation that the therapist can consider in their evaluation of the patient's progress. For the sake of patient privacy, the therapist will not have access to the user's diary entries. 
     
#### Progress:

  1. Sentiment Analyzing Model - training and integration into flask app complete.
  2. Summarizer - integration into flask app complete.
  3. Depression or Not (Binary Classifier) - training complete. Integration into flask app in progress.
  4. Facial Emotion Recognition Model - training complete. Integration into flask app in progress. 
 
#### To-Do (Tentatively):

  1. Improve UI/UX of flask app.
  2. Complete integration of models and database.
  3. Summary Visualizations on Therapist's side.
  4. Resources for managing depression on Patient's side.
  
