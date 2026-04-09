from pydantic import BaseModel, field_validator

EMPLOYMENT_TYPE_VALUES = ("FT", "CT", "PT", "FL")
EXPERIENCE_LEVEL_VALUES = ("MI", "SE", "EN", "EX")
REMOTE_RATIO_VALUES = (0, 50, 100)
WORK_YEAR_VALUES = (2020, 2021, 2022)
JOB_TITLE_CLEAN_VALUES = (
    "Data Scientist",
    "Machine Learning Scientist",
    "Big Data Engineer",
    "Machine Learning Engineer",
    "Data Analyst",
    "Business Data Analyst",
    "Lead Data Engineer",
    "Data Engineer",
    "Data Science Consultant",
    "BI Data Analyst",
    "Director of Data Science",
    "Research Scientist",
    "Data Engineering Manager",
    "ML Engineer",
    "AI Scientist",
    "Computer Vision Engineer",
    "Principal Data Scientist",
    "Data Science Manager",
    "Head of Data",
    "Data Analytics Engineer",
    "Applied Data Scientist",
    "Applied Machine Learning Scientist",
    "Data Analytics Manager",
    "Head of Data Science",
    "Data Architect",
    "Other",
)
EMPLOYEE_RESIDENCE_VALUES = (
    "DE",
    "JP",
    "GB",
    "HN",
    "US",
    "HU",
    "NZ",
    "FR",
    "IN",
    "PK",
    "PL",
    "PT",
    "CN",
    "GR",
    "AE",
    "NL",
    "MX",
    "CA",
    "AT",
    "NG",
    "PH",
    "ES",
    "DK",
    "RU",
    "IT",
    "HR",
    "BG",
    "SG",
    "BR",
    "IQ",
    "VN",
    "BE",
    "UA",
    "MT",
    "CL",
    "RO",
    "IR",
    "CO",
    "MD",
    "KE",
    "SI",
    "HK",
    "TR",
    "RS",
    "PR",
    "LU",
    "JE",
    "CZ",
    "AR",
    "DZ",
    "TN",
    "MY",
    "EE",
    "AU",
    "BO",
    "IE",
    "CH",
)
COMPANY_LOCATION_VALUES = (
    "DE",
    "JP",
    "GB",
    "HN",
    "US",
    "HU",
    "NZ",
    "FR",
    "IN",
    "PK",
    "CN",
    "GR",
    "AE",
    "NL",
    "MX",
    "CA",
    "AT",
    "NG",
    "ES",
    "PT",
    "DK",
    "IT",
    "HR",
    "LU",
    "PL",
    "SG",
    "RO",
    "IQ",
    "BR",
    "BE",
    "UA",
    "IL",
    "RU",
    "MT",
    "CL",
    "IR",
    "CO",
    "MD",
    "KE",
    "SI",
    "CH",
    "VN",
    "AS",
    "TR",
    "CZ",
    "DZ",
    "EE",
    "MY",
    "AU",
    "IE",
)
COMPANY_SIZE_VALUES = ("L", "S", "M")


class SalaryPredictionInput(BaseModel):
    work_year: int
    remote_ratio: int
    experience_level: str
    employment_type: str
    job_title_clean: str
    employee_residence: str
    company_location: str
    company_size: str

    @field_validator("work_year")
    @classmethod
    def validate_work_year(cls, value: int) -> int:
        if value not in WORK_YEAR_VALUES:
            raise ValueError(f"work_year must be one of {WORK_YEAR_VALUES}.")
        return value

    @field_validator("remote_ratio")
    @classmethod
    def validate_remote_ratio(cls, value: int) -> int:
        if value not in REMOTE_RATIO_VALUES:
            raise ValueError(f"remote_ratio must be one of {REMOTE_RATIO_VALUES}.")
        return value

    @field_validator("experience_level")
    @classmethod
    def validate_experience_level(cls, value: str) -> str:
        if value not in EXPERIENCE_LEVEL_VALUES:
            raise ValueError(f"experience_level must be one of {EXPERIENCE_LEVEL_VALUES}.")
        return value

    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, value: str) -> str:
        if value not in EMPLOYMENT_TYPE_VALUES:
            raise ValueError(f"employment_type must be one of {EMPLOYMENT_TYPE_VALUES}.")
        return value

    @field_validator("job_title_clean")
    @classmethod
    def validate_job_title_clean(cls, value: str) -> str:
        if value not in JOB_TITLE_CLEAN_VALUES:
            raise ValueError("job_title_clean must be one of the supported job titles.")
        return value

    @field_validator("employee_residence")
    @classmethod
    def validate_employee_residence(cls, value: str) -> str:
        if value not in EMPLOYEE_RESIDENCE_VALUES:
            raise ValueError("employee_residence must be one of the supported country codes.")
        return value

    @field_validator("company_location")
    @classmethod
    def validate_company_location(cls, value: str) -> str:
        if value not in COMPANY_LOCATION_VALUES:
            raise ValueError("company_location must be one of the supported country codes.")
        return value

    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, value: str) -> str:
        if value not in COMPANY_SIZE_VALUES:
            raise ValueError(f"company_size must be one of {COMPANY_SIZE_VALUES}.")
        return value


class SalaryPredictionResponse(BaseModel):
    predicted_salary_in_usd: float
