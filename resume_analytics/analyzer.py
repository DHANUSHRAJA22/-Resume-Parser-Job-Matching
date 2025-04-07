import spacy
from collections import Counter
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util # type: ignore

class ResumeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def analyze_resume(self, resume_text, job_description=None):
        doc = self.nlp(resume_text)
        word_count = len(resume_text.split())
        sentence_count = len(list(doc.sents))
        skills = self._extract_skills(doc)
        experience_years = self._analyze_experience(doc)
        profile_score = self._calculate_profile_score(word_count, sentence_count, len(skills), experience_years)
        result = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "skills_count": len(skills),
                "experience_years": experience_years,
                "profile_score": profile_score
            },
            "skills": list(skills),
            "suggestions": self._generate_suggestions(word_count, sentence_count, skills, experience_years)
        }
        if job_description:
            similarity = self.match_with_job(resume_text, job_description)
            result["job_match_score"] = similarity
        return result

    def _extract_skills(self, doc):
        entities = self.ner_pipeline(doc.text)
        skills = set()
        for ent in entities:
            if ent['entity_group'] in ['ORG', 'MISC', 'SKILL'] and len(ent['word']) > 2:
                skills.add(ent['word'].lower())
        return skills

    def _analyze_experience(self, doc):
        experience_years = 0
        for token in doc:
            if token.like_num and token.i < len(doc) - 1:
                next_token = doc[token.i + 1]
                if "year" in next_token.text.lower():
                    try:
                        experience_years = max(experience_years, int(token.text))
                    except ValueError:
                        continue
        return experience_years

    def _calculate_profile_score(self, word_count, sentence_count, skills_count, experience_years):
        score = 0
        score += 25 if word_count >= 300 else (word_count / 300) * 25
        score += 35 if skills_count >= 8 else (skills_count / 8) * 35
        score += 40 if experience_years >= 5 else (experience_years / 5) * 40
        return min(round(score), 100)

    def _generate_suggestions(self, word_count, sentence_count, skills, experience_years):
        suggestions = []
        if word_count < 300:
            suggestions.append({"icon": "fa-file-text", "text": "Add more detail to your resume - aim for at least 300 words"})
        if len(skills) < 8:
            suggestions.append({"icon": "fa-code", "text": "Include more relevant technical skills and technologies"})
        if sentence_count < 10:
            suggestions.append({"icon": "fa-list", "text": "Add more achievements and responsibilities from your experience"})
        if experience_years < 2:
            suggestions.append({"icon": "fa-briefcase", "text": "Highlight any internships, projects, or relevant coursework"})
        if not suggestions:
            suggestions.append({"icon": "fa-star", "text": "Your resume looks great! Consider adding more quantifiable achievements"})
        return suggestions

    def match_with_job(self, resume_text, job_description):
        resume_emb = self.embedder.encode(resume_text, convert_to_tensor=True)
        job_emb = self.embedder.encode(job_description, convert_to_tensor=True)
        return round(util.cos_sim(resume_emb, job_emb).item() * 100, 2)
