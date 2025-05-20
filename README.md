# 🧠 SITME — Sistema Integral de Tamizaje MuLtimodal para Endometriosis

SITME es una aplicación web fullstack construida con **Python (Flask)** en el backend y **HTML/CSS/JS** en el frontend. Utiliza un **dataset sintético** y una base de datos **PostgreSQL** alojada en Render para almacenar y gestionar registros clínicos simulados. Se mantiene activa gracias a **Uptime Robot**.

---

## 🛠️ Tecnologías Utilizadas

- 🐍 **Python + Flask** — API REST y lógica del backend  
  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python) 
  ![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?logo=flask)

- 🖼️ **HTML/CSS/JavaScript** — Interfaz de usuario  
  ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5) 
  ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3) 
  ![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow?logo=javascript)

- 🔒 **JWT Token / Bcrypt** — Login con encriptado  
  ![JWT](https://img.shields.io/badge/JWT-Auth-orange?logo=jsonwebtokens) 
  ![Bcrypt](https://img.shields.io/badge/Bcrypt-Password_Hashing-blueviolet)

- 🛢️ **PostgreSQL** — Base de datos relacional (hospedada en Render)  
  ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-blue?logo=postgresql) 
  ![Render](https://img.shields.io/badge/Hosted_on-Render-46d3ff?logo=render)

- 🔄 **UptimeRobot** — Monitoreo del servicio  
  ![UptimeRobot](https://img.shields.io/badge/Monitored_by-UptimeRobot-green)

- 🧪 **Dataset sintético** — Generación de registros clínicos  
  ![SyntheticData](https://img.shields.io/badge/Data-Synthetic_Clinical-orange)

---

## 🌐 Endpoints disponibles

| Método | Ruta                             | Descripción                                       |
|--------|----------------------------------|---------------------------------------------------|
| GET    | `/`                              | Devuelve la interfaz web principal (`index.html`) |
| POST   | `/api/login`                     | Permite el inicio de sesion                       |
| POST   | `/api/logout`                    | Permite el cierre de sesion                       |
| POST   | `/predict`                       | Realiza una predicción con los datos entregados   |
| POST   | `/generate_clinical_record`      | Genera un nuevo registro clínico sintético        |
| POST   | `/save_simulation`               | Guarda una simulación médica en la base de datos  |
| GET    | `/get_history`                   | Obtiene el historial completo de simulaciones     |
| DELETE | `/delete_record/<int:record_id>` | Elimina un registro clínico por su ID             |
| GET    | `/get_record_details/<int:id>`   | Obtiene detalles de un registro específico        |

---

## 🧭 Estado del proyecto

- ✅ Backend funcional
- ✅ Frontend operativo
- ✅ Login con tokens validados
- ✅ Conexión a PostgreSQL en Render
- ✅ Endpoints implementados
- ✅ Monitoreo con UptimeRobot

## 🧪 Créditos y agradecimientos

Este proyecto utiliza datos sintéticos para pruebas y desarrollo.
Creado por yvns_dev.
