# 🧠 SITME — Sistema Integral de Tamizaje para Endometriosis

SITME es una aplicación web fullstack construida con **Python (Flask)** en el backend y **HTML/CSS/JS** en el frontend. Utiliza un **dataset sintético** y una base de datos **PostgreSQL** alojada en Render para almacenar y gestionar registros clínicos simulados. Se mantiene activa gracias a **Uptime Robot**.

---

## 🚀 Tecnologías utilizadas

- 🐍 **Python** + **Flask** — API REST y lógica del backend
- 🖼️ **HTML/CSS/JavaScript** — Interfaz de usuario (frontend)
- 🔒 **JWT Token / Bcrypt** - Login con encriptado de contraseñas y tokens
- 🛢️ **PostgreSQL** — Base de datos relacional (hospedada en Render)
- 🔄 **UptimeRobot** — Mantiene el servicio activo en Render Free Tier
- 🧪 **Dataset sintético** — Generación de registros clínicos artificiales

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
