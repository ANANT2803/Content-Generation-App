import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

examples = [
    {
        "input": "Generate an Instagram content for the alumni achievement post for our alumnus, 'Mr. Pratyush Choudhury (BME '21)' who co-founded 'Activate VC' and raised '75 million USD' in funding" 
        "query": "SAIC is proud to share that Mr. Pratyush Choudhury (BME ‘21), alumnus of IIT (BHU), Varanasi, has co-founded Activate VC, India’s first AI-focused venture fund, with an impressive Fund I of $75 million, leveraging his exceptional trajectory in the venture ecosystem to become one of the youngest General Partners in the country, aiming to empower India’s next generation of AI founders, and we extend our heartfelt congratulations to him on this remarkable achievement."

    },
   {
       "input": "Generate an instagram content for the alumni achievement post for our alumnus, 'Mr.Prateek Maheshwari' for his role in the 'Physics Wallah' IPO."  
       "query": "SAIC is proud to celebrate this incredible achievement as we congratulate Prateek Maheshwari for being an important part of Physics Wallah’s spectacular IPO, highlighting his journey from campus corridors to contributing to one of India’s most inspiring ed-tech journeys, and here’s to innovation, impact, and many more milestones ahead."
   },
   {
       
      "input": "Generate a warm, invitation-style announcement post Instagram caption for 'Connect & Learn' happening in the 'Pink City' (Jaipur) this summer. The post should highlight the goal of providing clarity and connection for students navigating new challenges and emphasize that experiences will be shared and ideas challenged. Include all specified event details (Date: 11 July, Friday,Time: 6:00 PM Onwards)",
      "query": "Still figuring things out? You’re not alone. New cities, new skills, new questions—it’s all part of the ride. That’s where Connect & Learn steps in. This summer, SAIC is bringing students and alumni together in the Pink City—where centuries of history echo through bold, new voices and stories flow as easily as sand. It’s where experiences are unpacked, ideas are challenged, and every interaction has the potential to shape what comes next. If you are seeking connection, clarity, a sense of direction or just the comfort of being understood, this is the place for you. Event Details: Date: 11 July, Friday Time: 6:00 PM Onwards Venue: https://maps.app.goo.gl/Zei5JHDz9ox8Cw989?g_st=ipc\n\nSeats are limited. Register now through the link in our bio."

   },
   {
       
     "input": "Generate an alumni achievement post for alumni Dirghayu Kaushik and Vikrant Shivalik, founders of Ambitio (AI study abroad platform). Mention their $2M funding and mission to transform global education."
     "query": "Built on friendship, fueled by resilience, What started as a dream is now shaping the future of global education! SAIC is proud to share the success of Mr. Dirghayu Kaushik (PHE ‘21), Founder & CEO, and Mr. Vikrant Shivalik (PHE ‘21), Founder & COO. They are the visionaries behind Ambitio, an AI-powered platform transforming the study abroad experience, making global education more accessible and seamless. Ambitio has secured $2 million in funding to drive this revolution. We wish them continued success in transforming the future of study abroad! "

   },
   {
    
    "input": "Generate an alumni achievement post from SAIC for alumnus Mr. Siddharth Jain (IIT BHU) on his appointment as Managing Partner and Country Head of Kearney. Highlight his 17 years of consulting experience, focus on operational excellence/long-term success, and his impact across India, SEA, ME, and Africa.",
    "query": "SAIC is elated to share that Mr. Siddharth Jain, an alumnus of IIT (BHU), Varanasi has been appointed as Managing Partner and Country Head of Kearney, a leading global management consulting firm. Mr. Jain provides over 17 years of excellent consulting experience, constantly excelling at increasing operational excellence and creating long-term success. His strategic acumen and significant achievements have echoed well beyond India, reaching Southeast Asia, the Middle East, and Africa. His incredible path demonstrates the value of endurance, vision, and leadership, serving as a beacon for our students and alumni. We extend our warmest congratulations to him on this remarkable milestone and wish him continued success in driving meaningful impact across industries and communities."

   },
   {
       
    "input": "Generate a congratulatory post from SAIC for alumna Ms. Gayatri Bandi (MIN ’22) for being named to the WIM UK 2024 '100 Global Inspirational Women In Mining' list. Highlight her distinction as the first female underground mining engineer in the 117-year history of the Jharia coalfields. Mention her advocacy for inclusivity as the head of the Women of MEAI.",
    "query": "SAIC proudly announces that Ms. Gayatri Bandi (MIN ’22), a pioneer in mining and inclusivity, has been recognized on the Women In Mining UK (WIM UK) 2024 list of ‘100 Global Inspirational Women In Mining.’ She serves as the Manager in Tata Steel’s Mining Division and is the first female underground mining engineer in the 117-year history of the Jharia coalfields. A recipient of the Roberton Medal, she has led vital projects in a male-dominated field. Additionally, Ms. Gayatri passionately advocates for inclusivity as the head of the Women of MEAI (Mining Engineers Association of India). This prestigious recognition highlights the exceptional contributions of women in the mining industry."

   },
   {
       
     "input": "Generate a Mail for IIT (BHU) alumni to register as mentors for SAMP 2026, highlighting the impact of the 2025 cycle (200+ mentors, 800+ students) and the goal of bridging the gap between professional experience and student aspirations with a clear call to action."
     "query": "The Student Alumni Interaction Cell (SAIC) at IIT (BHU) Varanasi invites alumni to join the Student Alumni Mentorship Program (SAMP) 2026, an initiative that leverages seasoned professional experience to guide current students across diverse domains; building on the success of 2025—which saw 200+ mentors impacting over 800 students—this program offers a unique opportunity for alumni to turn their past experiences into stepping stones for future graduates and renew their bond with the Institute."

   },
   {
       
     "input":"Generate a LinkedIn post for SAIC’s Student Alumni Mentorship Program for 2nd-year and above students. Mention the 200+ alumni mentors across diverse domains (SDE, AI/ML, Core, etc.), the success of last year's 800+ mentees, and include a call to action for registration before the deadline."
     "query":"SAIC is back with the flagship Student Alumni Mentorship Program, an initiative that pairs 2nd-year and above students with 200+ experienced alumni mentors across domains like SDE, AI/ML, Product Management, and Core Research to bridge the gap between academic roadmaps and professional excellence; following a successful year with 800+ mentees, this program offers high-impact, tailored guidance and the opportunity to build lasting relationships that unlock true professional potential."
   
   },
   {
       
     "input": "Generate a catchy and informative WhatsApp message for SAIC’s Student Alumni Mentorship Program for students in 2nd year and above, to highlight key stats like 200+ mentors and 800+ past mentees, and emphasizing the January 15th deadline with the registration link."
     "query": "SAIC is thrilled to announce the opening of registrations for the Student Alumni Mentorship Program, a flagship initiative where 'Experience guides and Ambition finds direction' by pairing students in their 2nd year or above with over 200+ expert alumni mentors across diverse domains like SDE, AI/ML, and Core sectors; following the incredible impact of our previous edition which saw 800+ mentees benefit from one-on-one professional guidance, this is your chance to gain a competitive edge and build lasting career networks, so ensure you register at before the deadline on 15th January 2026 to start your journey toward leadership."
   
   },
   {
       
     "input":"Generate an engaging Instagram caption for SAIC’s Student Alumni Mentorship Program 2026, targeting 2nd-year and above students. Mention last year stats of 200+ mentors and 800+ mentees, and the value of one-on-one professional guidance across diverse domains, including a clear call to action to check the registration link in the bio."
     "query":"SAIC is delighted to open registrations for the Student Alumni Mentorship Program 2026, an initiative where experience guides and ambition finds direction by facilitating one-on-one interactions between students and our distinguished alumni network; following a highly successful previous edition that empowered over 800 mentees through the guidance of 200+ mentors across diverse professional domains, we invite all students in their second year or above to seize this opportunity for personalized career insights and academic growth by registering via the link in our bio before the upcoming deadline."
   
   },
   {
       
     "input":"Generate a high-energy Instagram caption for SAIC’s 'Beyond The Degree' session featuring IIT (BHU) alumnus Yash Shukla, focusing on his journey from MET’20 to IIM Ahmedabad. Highlight the insider perspective on MBA interviews and global project experience at Applied Materials, include the session details for January 3rd at 1:00 PM with a joining link, and use engaging emojis to attract students interested in MBA prep and career growth."
     "query":"SAIC is proud to present an exclusive 'Beyond The Degree' session featuring Mr. Yash Shukla, an IIT (BHU) alumnus and current IIM Ahmedabad student, who will pull back the curtain on the MBA interview process and share first-hand insights from his journey from the MET’20 batch to leading global projects at Applied Materials; this session is a must-attend for anyone curious about what really happens inside the interview room, offering practical preparation tips and a live Q&A starting at 1:00 PM on 3rd January 2026 via the provided link, so don't miss this opportunity to learn from a leader who has successfully navigated the path to India's premier B-school."
  
   },
   {
       
     "input":"Generate an engaging WhatsApp broadcast for SAIC’s 'Beyond The Degree' session featuring Yash Shukla, focusing on his transition from IIT (BHU) MET'20 to IIM Ahmedabad and his professional experience at Applied Materials. Emphasize the unique opportunity to learn about MBA interviews and preparation tips first-hand, including the event details for January 3rd at 1:00 PM and the joining link, using formatting and emojis suitable for student groups."
     "query":"SAIC invites you to an exclusive 'Beyond The Degree' session with Mr. Yash Shukla, an IIT (BHU) MET'20 alumnus and current IIM Ahmedabad student, who will share his firsthand experiences and prep strategies for the journey from engineering to a top-tier MBA; featuring insights from his time leading global projects at Applied Materials, this session on January 3rd at 1:00 PM via the provided link offers a rare look at what happens inside the interview room and provides a platform for students to get their questions answered by someone who has successfully navigated the elite B-school selection process."
  
   },
   {
       
     "input":"Generate a professional LinkedIn post for SAIC’s 'Beyond The Degree' panel discussion featuring Yash Shukla and Prakhar Singh, both IIT (BHU) alumni now at IIM Ahmedabad. Highlight the session's focus on 'Cracking MBA Interviews,' covering profile presentation, impactful answer structuring, and interviewer expectations, while including event details for January 4th at 12:45 PM and a call to action to join via the link."
     "query":"SAIC is proud to announce a high-impact panel discussion under its Beyond The Degree initiative titled 'Cracking MBA Interviews,' featuring the combined expertise of Yash Shukla (MET ’20, IIM A ’27) and Prakhar Singh (EEE ’25, IIM A ’27) as they share their firsthand success stories and strategies for navigating the elite B-school selection process; this comprehensive session will delve into the nuances of structuring impactful interview answers, effectively presenting academic and professional profiles, and understanding the core attributes that top-tier interview panels prioritize, offering students a unique opportunity to engage in a live Q&A and gain practical takeaways for their own MBA journeys on January 4th at 12:45 PM via the provided session link."
   
   },
   {
       
     "input":"Generate a professional LinkedIn post caption for SAIC’s Alumni-Guided Mock Interviews Placement Drive, inviting both students and alumni to participate. Highlight the initiative's goal of bridging the gap between academic learning and real-world recruitment through one-on-one mentorship, personalized feedback, and exposure to diverse domain interview styles, while including the registration links for both groups and the October 15th deadline."
     "query":"SAIC is proud to launch the Alumni-Guided Mock Interviews Placement Drive, a strategic initiative designed to empower students by connecting them with our esteemed alumni network for realistic interview simulations that provide invaluable exposure to diverse domain styles and recruiter expectations; this program offers a unique platform for students to refine their interview techniques and for alumni to give back by sharing professional expertise and personalized feedback, ensuring that our future graduates transition from academic excellence to professional success with confidence—interested alumni and students should register via the provided links before the October 15, 2025 deadline to participate in this impactful journey of career preparation."
   
   },
   {
       
     "input":"Generate an engaging Instagram caption for SAIC’s Alumni-Guided Mock Interviews placement edition, building on the success of the previous internship drive. Focus on the value of personalized resume feedback and firsthand alumni insights for placement prep, while inviting alumni to continue their support and urging students to register via the link in the bio before the specified deadline."
     "query":"Following the incredible success and impact of the internship edition, SAIC is thrilled to bring back the Alumni-Guided Mock Interviews specifically for the placement season to help students sharpen their interview skills and refine their resumes through one-on-one sessions with our experienced alumni network; this initiative remains a cornerstone of student preparation, offering a unique opportunity to gain the competitive edge needed for placements while fostering a strong culture of mentorship within our community, so we invite all students to secure their spot by filling out the form in our bio and encourage our esteemed alumni to join us once again in shaping the career paths of the next batch of leaders before the registration deadline."
   
   },
   {
       
     "input":"Generate a motivating and concise WhatsApp message for SAIC’s Alumni-Guided Mock Interviews placement edition, emphasizing the transition from interview jitters to offer letters. Highlight the benefits of a real interview environment, seasoned alumni insights, and personalized resume feedback, while including a call to action to register via the provided link."
     "query":"SAIC is back with the placement edition of Alumni-Guided Mock Interviews, a dedicated initiative to help students transform interview jitters into official offer letters by providing a realistic simulation of the recruitment process; this program offers a golden opportunity to interact with seasoned alumni, gain critical insights into industry expectations, and receive tailored feedback on both your performance and resume to ensure you walk into your final placements with absolute confidence, so make sure to register via the link provided and take this essential step toward acing your career goals."
   
   },
   {
       
     "input":"Generate an inviting and concise WhatsApp message for SAIC’s 'Connect & Learn: Winter Edition – Hyderabad' student-alumni meetup. Emphasize the opportunity for networking, learning from the lived experiences of alumni, and gaining practical insights in an informal setting, while highlighting the event details for December 21st and the registration link."
     "query":"SAIC is excited to announce the Winter Edition of Connect & Learn in Hyderabad, a premier student-alumni meetup designed to bridge the gap between campus and the professional world through direct interaction and shared experiences; this event offers a unique platform to gain practical career insights and network with alumni who have successfully navigated the paths you are currently on, so don't miss this chance to connect in person on 21st December 2025 starting at 11:30 AM by registering at https://tinyurl.com/cnlhyderabad to secure your spot for an afternoon of mentorship and meaningful community building."
   
   },
   {
       
      "input":"Generate a high-impact Instagram caption for SAIC’s 'Connect & Learn: Winter Edition – Hyderabad,' building on the success of the 7-city summer tour. Focus on the emotional journey of navigating career crossroads, the value of unscripted advice from alumni who have been there, and the goal of providing clarity through real stories and meaningful connections, including the event details for December 21st and a call to action for the registration link in the bio."
      "query":"SAIC is bringing its celebrated 'Connect & Learn' initiative to Hyderabad this winter, following a phenomenal summer edition that spanned seven cities, to provide a dedicated space for students who find themselves at the crossroads of career decisions and future planning; moving beyond scripted professional advice, this meetup focuses on honest, lived experiences and real stories from alumni who have navigated the same transitions, offering students a unique opportunity to gain clarity and reassurance through meaningful community connections on 21st December 2025 starting at 11:30 AM—so if you're looking for practical perspectives to help figure out your next big move, register via the link in our bio to join us for this highly anticipated event."
   
   },
   {
       
     "input":"Generate a professional and inspiring LinkedIn post for SAIC’s 'Connect & Learn: Winter Edition – Hyderabad,' highlighting the value of real-world alumni experiences over traditional classroom lessons. Emphasize the diversity of sectors represented—from Tech and Consulting to Core Engineering—and the success of the previous 7-city summer tour, while inviting students to gain career clarity and network with local alumni on December 21st."
     "query":"Building on the momentum of a successful summer edition that spanned seven cities, SAIC is proud to bring 'Connect & Learn' to Hyderabad this winter, offering a premier platform for IIT (BHU) students to supplement their academic learning with real-world insights from alumni leaders across diverse sectors including technology, consulting, management, and core engineering; this meetup is designed to facilitate meaningful mentorship and career guidance in a city known for its innovation, allowing students to learn firsthand from those who have successfully navigated career pivots and professional uncertainties, so join us on December 21st, 2025, starting at 11:30 AM to build the connections that will shape your professional future and provide the clarity needed to excel in today's rapidly evolving job market."
   
   }
   

]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

# Embeddings Config
OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
OPENAI_API_VERSION_EMBEDDING = os.getenv("OPENAI_API_VERSION_EMBEDDING")

@st.cache_resource
def get_example_selector():
    """
    Creates a smart selector that finds the most relevant examples 
    from the list above based on the semantic meaning of the user's input.
    """
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=OPENAI_EMBEDDINGS_MODEL_NAME
        ),
        Chroma, # Vector Store
        k=1, # We only need 1 or 2 good examples for style matching
        input_keys=["input"],
    )
    return example_selector
